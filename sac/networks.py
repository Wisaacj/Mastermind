import os
import torch as T
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.distributions.normal import Normal
from typing import Tuple


class CriticNetwork(nn.Module):

    def __init__(
        self,
        beta: float, 
        input_dims: Tuple, 
        n_actions: int, 
        fc1_dims: int = 256, 
        fc2_dims: int = 256,
        name: str = 'critic',
        chkpt_dir: str = 'tmp/sac'
    ):
        """
        
        :param beta: the learning rate.
        """
        super().__init__()

        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        if not os.path.exists(chkpt_dir):
            os.mkdir(chkpt_dir)

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        # Critic evaluate the value of a state and action pair.
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Output is a scalar quantity.
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimiser = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        # First FC layer. dim=1 is the batch dimension
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        # Second FC layer.
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        return self.q(action_value)
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    """
    Value network just estimates the value of a particular state or set of states. It
    doesn't care which action you took or are taking.
    """

    def __init__(
        self,
        beta: float,
        input_dims: Tuple,
        fc1_dims: int = 256,
        fc2_dims: int = 256,
        name: str = 'value',
        chkpt_dir: str = 'tmp/sac',
    ):
        super().__init__()

        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name

        if not os.path.exists(chkpt_dir):
            os.mkdir(chkpt_dir)

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)
        
        self.optimiser = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        return self.v(state_value)
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    """
    This network is arguably more complicated because you have to handle sampling
    a probability distribution instead of just sampling a feed-forward network.
    """

    def __init__(
        self,
        alpha: float,
        input_dims: Tuple,
        max_action: int | float,
        fc1_dims: int = 256,
        fc2_dims: int = 256,
        n_actions: int = 2,
        name: str = 'actor',
        chkpt_dir: str = 'tmp/sac'
    ):
        """
        
        :param alpha: the learning rate.
        :param max_action: the purpose of max action is that our choose action function
            is going to be restricted to +- 1 with a tanh function, but an environemnt
            may very well have an action bound much greater than this. Thus, you want the
            output of the DNN of the sampling function to be multiplied by the max action.
        """
        super().__init__()

        self.alpha = alpha
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action
        # This serves a number of functions. Firstly, it makes sure we don't take the
        # log of 0, which is undefined.
        self.reparam_noise = 1e-6

        self.name = name

        if not os.path.exists(chkpt_dir):
            os.mkdir(chkpt_dir)

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Mean and standard deviation for each action.
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)

        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        # We clamp sigma because the don't want the distribution for our policy to be
        # arbitrarily broad. The standard deviation defines the width of your distribution.
        # You could also use a sigmoid activation function here, but that is a little
        # slower in terms of computational speed and clamp is much faster.
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma
    
    def sample_normal(self, state, reparameterise: bool = True):
        """
        This method enables the actor-critic model to handle continuous action spaces.
        """
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterise:
            # This adds some additional noise, i.e., some additional exploration factor
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        # Apply hyperbolic tan activation function.
        action = T.tanh(actions) # .to(self.device)

        # This is for the calculation of our loss function. It isn't a component
        # of which action to take.
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        # Scale action value.
        action = action * T.tensor(self.max_action).to(self.device)
        
        return action, log_probs
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
