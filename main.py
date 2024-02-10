import gymnasium as gym
import numpy as np

from sac.sac import SoftActorCritic
from utils import plot_learning_curve


if __name__ == "__main__":
    load_checkpoint = False

    if load_checkpoint:
        env = gym.make("InvertedPendulum-v4", render_mode="human")
    else:
        env = gym.make("InvertedPendulum-v4")

    agent = SoftActorCritic(
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0]
    )
    n_games = 350

    filename = 'inverted_pendulum.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation, _ = env.reset()
        done = truncated = False
        score = 0

        while not (done or truncated):
            action = agent.choose_action(observation)
            obs_, reward, done, truncated, info = env.step(action)
            score += reward
            done = done or truncated
            agent.remember(observation, action, reward, obs_, done)

            if not load_checkpoint:
                agent.learn()

            observation = obs_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

            if not load_checkpoint:
                agent.save_models()

        print('episode %d,' % i, 'score %.1f,' % score, 'avg_score %.1f,' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)