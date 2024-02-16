import gymnasium as gym
import numpy as np
import torch
import random
import pandas as pd

from collections import deque
from sac.sac import DiscreteSoftActorCritic
from utils import plot_learning_curve
from environment import Mastermind


def main():
    n_episodes = 750
    seed = 1
    warmup_steps = 10_000
    load_checkpoint = False

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    # env = Mastermind(code_length=3, num_colours=3)
    env.action_space.seed(seed)

    steps = 0

    agent = DiscreteSoftActorCritic(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape or (1,),
        n_actions=env.action_space.n,
        env=env,
        warmup_steps=warmup_steps,
        tau=0.005
    )

    filename = 'cartpole_2500.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_episodes):
        state, _ = env.reset()
        episode_steps = 0
        rewards = 0

        done = truncated = False

        while not (done or truncated):
            action = agent.get_action(state)
            steps += 1

            next_state, reward, done, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done or truncated)

            policy_loss, _, _, _, _ = agent.learn()
            state = next_state
            rewards += reward
            episode_steps += 1

        score_history.append(rewards)
        # avg_score = np.mean(score_history[-100:])
        if i % 1 == 0:
            print("Episode: {} | Reward: {} | Policy Loss: {} | Steps: {}"\
                .format(i, rewards, policy_loss, steps,))
        
    if not load_checkpoint:
        x = [i+1 for i in range(n_episodes)]
        plot_learning_curve(x, score_history, figure_file)

    # Save results to disk as a csv file.
    pd.DataFrame({"SAC": score_history}).to_csv("./results/cartpole/sac2.csv")


if __name__ == "__main__":
    main()