import gymnasium as gym
import numpy as np
from collections import deque
import torch
from sac.buffer import ReplayBuffer
import random
from sac.sac import SACDiscrete


def collect_random(env: gym.Env, dataset: ReplayBuffer, num_samples=200):
    state, _ = env.reset()

    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        dataset.store_transition(state, action, reward, next_state, done or truncated)
        state = next_state

        if done or truncated:
            state, _ = env.reset()


if __name__ == "__main__":
    n_episodes = 1000
    seed = 42
    buffer_size = 1000000
    batch_size = 256

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0
        
    agent = SACDiscrete(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )

    buffer = ReplayBuffer(buffer_size, env.observation_space.shape, (1,))
    
    collect_random(env=env, dataset=buffer, num_samples=10000)
    
    for i in range(1, n_episodes+1):
        state, _ = env.reset()
        episode_steps = 0
        rewards = 0
        while True:
            action = agent.get_action(state)
            steps += 1
            next_state, reward, done, truncated, _ = env.step(action)
            buffer.store_transition(state, action, reward, next_state, done or truncated)
            policy_loss, _, _, _, _ = agent.learn(steps, buffer.sample_buffer(batch_size), gamma=0.99)
            state = next_state
            rewards += reward
            episode_steps += 1

            if done or truncated:
                break

        average10.append(rewards)
        total_steps += episode_steps
        print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps,))