import os
import torch as T
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from typing import Tuple


class SavableNetwork(nn.Module):

    def __init__(
        self,
        name: str,
        chkpt_dir: str,
        seed: int,
    ):
        super().__init__()

        if not os.path.exists(chkpt_dir):
            os.mkdir(chkpt_dir)

        self.name = name
        self.seed = T.manual_seed(seed)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(SavableNetwork):

    def __init__(
        self,
        beta: float, 
        input_dims: Tuple, 
        n_actions: int, 
        fc1_dims: int = 256, 
        fc2_dims: int = 256,
        name: str = 'critic',
        chkpt_dir: str = 'tmp/sac',
        seed: int = 42,
    ):
        """
        
        :param beta: the learning rate.
        """
        super().__init__(name, chkpt_dir, seed)

        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

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


class ValueNetwork(SavableNetwork):
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
        super().__init__(name, chkpt_dir)

        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

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


class ActorNetwork(SavableNetwork):
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
        super().__init__(name, chkpt_dir)

        self.alpha = alpha
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action
        # This serves a number of functions. Firstly, it makes sure we don't take the
        # log of 0, which is undefined.
        self.reparam_noise = 1e-6

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
        sigma = T.clamp(sigma, min=-20, max=2)
        # sigma = F.softplus(sigma) + self.reparam_noise

        return mu, sigma
    
    def sample_normal(self, state, reparameterise: bool = True):
        """
        This method enables the actor-critic model to handle continuous action spaces.
        """
        mu, sigma = self.forward(state)
        sigma = sigma.exp()

        probabilities = Normal(mu, sigma)

        if reparameterise:
            # This adds some additional noise, i.e., some additional exploration factor
            x_t = probabilities.rsample()
        else:
            x_t = probabilities.sample()

        max_action = T.tensor(self.max_action).to(self.device)

        # Apply hyperbolic tan activation function.
        y_t = T.tanh(x_t)
        # Scale action.
        action = y_t * max_action

        # This is for the calculation of our loss function. It isn't a component
        # of which action to take.
        log_probs = probabilities.log_prob(x_t)

        # Enforcing action bound.
        log_probs -= T.log((1 - y_t.pow(2)) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs
    

def hidden_init(layer: nn.Linear):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
    

class DiscreteCriticNetwork(SavableNetwork):

    def __init__(
        self,
        learning_rate: float, 
        state_size: int, 
        action_size: int, 
        fc1_dims: int = 256, 
        fc2_dims: int = 256,
        name: str = 'critic',
        chkpt_dir: str = 'tmp/sac',
        seed: int = 1,
    ):
        super().__init__(name, chkpt_dir, seed)

        self.learning_rate = learning_rate
        self.state_size = state_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_size = action_size

        # Critics evaluate the value of a state.
        self.fc1 = nn.Linear(state_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Output is a scalar quantity.
        self.q = nn.Linear(self.fc2_dims, action_size)
        self.reset_parameters()

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.q.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.q(x)
    

class DiscreteActorNetwork(SavableNetwork):

    def __init__(
        self,
        learning_rate: float,
        state_size: int,
        action_size: int,
        fc1_dims: int = 256,
        fc2_dims: int = 256,
        name: str = 'discrete_actor',
        chkpt_dir: str = 'tmp/sac',
        seed: int = 1,
    ):
        super().__init__(name, chkpt_dir, seed)

        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(state_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, action_size)
        self.softmax = nn.Softmax(dim=-1)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        action_probs = self.softmax(self.fc3(x))
        return action_probs
    
    def evaluate(self, state):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(self.device)

        # We have to deal with the situation of 0.0 probabilities because log(0) is
        # undefined.
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = T.log(action_probs + z)

        return action.detach().cpu(), action_probs, log_action_probs
    
    def get_action(self, state):
        return self.evaluate(state)

    def get_det_action(self, state):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(self.device)

        return action.detach().cpu()
