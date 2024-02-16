import os
import copy
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import gymnasium as gym

from typing import Tuple
from torch.nn.utils import clip_grad_norm_
from sac.buffer import ReplayBuffer
from sac.networks import ActorNetwork, ValueNetwork, CriticNetwork, DiscreteActorNetwork, DiscreteCriticNetwork


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


class DiscreteSoftActorCritic(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
        state_shape: Tuple,
        action_shape: Tuple,
        n_actions: int,
        env: gym.Env,
        warmup_steps: int,
        tau: float = 1e-2,
        gamma: float = 0.99,
        learning_rate: float = 5e-4,
        clip_grad_param: float = 1,
        mem_size: int = 1_000_000,
        batch_size: int = 256,
        seed: int = 1,
    ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super().__init__()
        
        self.env = env
        self.tau = tau
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.clip_grad_param = clip_grad_param
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.state_size = np.prod(state_shape)
        self.n_actions = n_actions
        
        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_size, state_shape, action_shape)
        self._collect_random_experience(warmup_steps)

        # Target entropy is -dim(Action Space)
        self.target_entropy = -self.n_actions

        self.log_alpha = T.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
                
        # Actor network 
        self.actor_local = DiscreteActorNetwork(
            learning_rate, self.state_size, self.n_actions).to(self.device)
        
        # Behavioural critic networks
        self.critic1 = DiscreteCriticNetwork(
            learning_rate, self.state_size, self.n_actions, seed=seed+1).to(self.device)
        self.critic2 = DiscreteCriticNetwork(
            learning_rate, self.state_size, self.n_actions, seed=seed).to(self.device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        # Target critic networks
        self.critic1_target = DiscreteCriticNetwork(
            learning_rate, self.state_size, self.n_actions, seed=seed).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = DiscreteCriticNetwork(
            learning_rate, self.state_size, self.n_actions, seed=seed).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def get_action(self, state: np.ndarray):
        """Returns actions for given state as per current policy."""
        state = T.from_numpy(state.flatten()).float().to(self.device)
        
        with T.no_grad():
            action = self.actor_local.get_det_action(state)

        return action.numpy()
    
    def learn(self):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        actions = T.from_numpy(actions).float().to(self.device)
        rewards = T.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        dones = T.from_numpy(dones.astype(np.uint8)).float().unsqueeze(1).to(self.device)
        states = T.from_numpy(states.reshape((self.batch_size, -1))).float().to(self.device)
        next_states = T.from_numpy(next_states.reshape((self.batch_size, -1))).float().to(self.device)

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self._calc_policy_loss(states, current_alpha.to(self.device))
        self.actor_local.optimiser.zero_grad()
        actor_loss.backward()
        self.actor_local.optimiser.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with T.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (T.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1)) 

        # Compute critic loss
        q1 = self.critic1(states).gather(1, actions.long())
        q2 = self.critic2(states).gather(1, actions.long())
        
        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

        # Update critics
        # critic 1
        self.critic1.optimiser.zero_grad()
        critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1.optimiser.step()
        # critic 2
        self.critic2.optimiser.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2.optimiser.step()

        # ----------------------- update target networks ----------------------- #
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), current_alpha

    def _soft_update(self, local_model: nn.Module, target_model: nn.Module):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def _calc_policy_loss(self, states, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states)   
        q2 = self.critic2(states)
        min_Q = T.min(q1,q2)
        actor_loss = (action_probs * (alpha * log_pis - min_Q )).sum(1).mean()
        log_action_pi = T.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi
    
    def _collect_random_experience(self, num_samples: int):
        state, _ = self.env.reset()

        for _ in range(num_samples):
            action = self.env.action_space.sample()
            next_state, reward, done, truncated, _ = self.env.step(action)
            self.remember(state, action, reward, next_state, done or truncated)
            state = next_state

            if done or truncated:
                state, _ = self.env.reset()


class LegDiscreteSoftActorCritic:

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