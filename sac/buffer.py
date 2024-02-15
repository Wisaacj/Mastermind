import numpy as np

from typing import Tuple, Dict


class ReplayBuffer:

    def __init__(self, max_size: int, state_shape: Tuple, action_shape: Tuple):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *state_shape))
        self.new_state_memory = np.zeros((self.mem_size, *state_shape))
        self.action_memory = np.zeros((self.mem_size, *action_shape))
        self.reward_memory = np.zeros(self.mem_size)
        # Reason we have this is becasue the value for the terminal state is identically
        # zero. We store the done flags from the environment as a way of setting the values
        # for this terminal state to zero.
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones
