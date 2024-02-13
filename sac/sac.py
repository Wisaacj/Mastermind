import os
import torch as T
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from sac.buffer import ReplayBuffer
from sac.networks import ActorNetwork, ValueNetwork, CriticNetwork, DiscreteActorNetwork


class SoftActorCritic:

    def __init__(
        self,
        alpha: float = 0.0003,
        beta: float = 0.0003,
        input_dims: Tuple = (8,),
        env = None,
        gamma: float = 0.99,
        n_actions: int = 2,
        max_size: int = 1000000,
        tau: float = 0.005,
        layer1_size: int = 256,
        layer2_size: int = 256,
        batch_size: int = 256,
        reward_scale: int = 2,
    ):
        """
        
        :param tau: this is use to perform a soft copy of the behavioural network onto
            the target network. It is a mixing parameter.
        """
        self.tau = tau
        self.gamma = gamma

        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.actor = ActorNetwork(alpha, input_dims, max_action=env.action_space.high, n_actions=n_actions)

        # There are two critic networks. We take the minimum of the evaluation of the two.
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions, name='critic_2')

        # As SAC is off-policy, we use both a behavioural and target value network.
        self.value = ValueNetwork(beta, input_dims)
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        # We apply `unsqueeze` to add a batch dimension.
        state = T.tensor(observation, dtype=T.float).unsqueeze(0).to(self.device)
        actions, _ = self.actor.sample_normal(state, reparameterise=False)

        return actions.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau: float = None):
        """
        
        :param tau: when `tau` = 1, this method performs a hard update of the target
            network parameters.
        """
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            # Here, we mix the parameter values from the behavioural and target networks.
            target_value_state_dict[name] = tau * value_state_dict[name].clone() + \
                (1 - tau) * target_value_state_dict[name]
            
        self.target_value.load_state_dict(target_value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)

        # Predict the value of the states.
        value = self.value(state).view(-1)
        # Value of the successor state, according to the target network.
        value_ = self.target_value(new_state).view(-1)

        # Terminal states should be zero-valued.
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterise=False)
        log_probs = log_probs.view(-1)

        # The critic values are the Q-values. We take the minimum of both critic
        # networks, which improves the stability of learning. In Deep Q-Learning there
        # is the problem of the overestimation bias which is a consequence of using a
        # max operator over actions in the q-learning update rule and a consequence
        # of using deep neural networks.
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # Backpropagate value network loss.
        self.value.optimiser.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimiser.step()

        # Backpropagate actor network loss.
        actions, log_probs = self.actor.sample_normal(state, reparameterise=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimiser.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimiser.step()

        # Backpropagate critic network loss.
        self.critic_1.optimiser.zero_grad()
        self.critic_2.optimiser.zero_grad()

        # Is this the TD-Target?
        q_hat = self.scale * reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)

        critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic_1.optimiser.step()
        self.critic_2.optimiser.step()

        self.update_network_parameters()


class DiscreteSoftActorCritic:

    def __init__(
        self,
        state_shape: Tuple,
        n_actions: int,
        alpha: float = 0.0003,
        beta: float = 0.0003,
        gamma: float = 0.99,
        max_size: int = 1000000,
        tau: float = 0.005,
        layer1_size: int = 256,
        layer2_size: int = 256,
        batch_size: int = 256,
        reward_scale: int = 1,
    ):
        """
        
        :param tau: this is use to perform a soft copy of the behavioural network onto
            the target network. It is a mixing parameter.
        """
        self.tau = tau
        self.gamma = gamma

        self.n_actions = n_actions
        self.batch_size = batch_size
        self.input_dims = (np.prod(state_shape),)
        self.memory = ReplayBuffer(max_size, self.input_dims, 1)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.actor = DiscreteActorNetwork(alpha, self.input_dims, n_actions=n_actions)

        # There are two critic networks. We take the minimum of the evaluation of the two.
        self.critic_1 = CriticNetwork(beta, self.input_dims, 1, name='critic_1')
        self.critic_2 = CriticNetwork(beta, self.input_dims, 1, name='critic_2')

        # As SAC is off-policy, we use both a behavioural and target value network.
        self.value = ValueNetwork(beta, self.input_dims)
        self.target_value = ValueNetwork(beta, self.input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        # We apply `unsqueeze` to add a batch dimension.
        state = T.tensor(observation, dtype=T.float).unsqueeze(0).to(self.device)
        # probabilities = self.actor(state)
        # action = T.argmax(probabilities).item()

        action, _ = self.actor.sample_discrete(state)

        return action.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau: float = None):
        """
        
        :param tau: when `tau` = 1, this method performs a hard update of the target
            network parameters.
        """
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            # The parameters of target network are updated here.
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.int).to(self.device)

        # Predict the value of the states.
        value = self.value(state).view(-1)
        # Value of the successor state, according to the target network.
        value_ = self.target_value(new_state).view(-1)

        # Terminal states should be zero-valued.
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_discrete(state)
        actions = actions.unsqueeze(1)
        log_probs = log_probs.view(-1)

        # The critic values are the Q-values. We take the minimum of both critic
        # networks, which improves the stability of learning. In Deep Q-Learning there
        # is the problem of the overestimation bias which is a consequence of using a
        # max operator over actions in the q-learning update rule and a consequence
        # of using deep neural networks.
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # Backpropagate value network loss.
        self.value.optimiser.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimiser.step()

        # Backpropagate actor network loss.
        actions, log_probs = self.actor.sample_discrete(state)
        actions = actions.unsqueeze(1)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = -(log_probs - critic_value).mean()
        self.actor.optimiser.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimiser.step()

        # Backpropagate critic network loss.
        self.critic_1.optimiser.zero_grad()
        self.critic_2.optimiser.zero_grad()

        # Is this the TD-Target?
        q_hat = self.scale * reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)

        critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic_1.optimiser.step()
        self.critic_2.optimiser.step()

        self.update_network_parameters()