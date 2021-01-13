import itertools
import sys
from collections import defaultdict

import numpy as np

from envs.windy_gridworld import WindyGridworldEnv
from lib.utils import make_epsilon_greedy_policy, EpisodeStats
from lib import plotting
from td import TemporalDifference

""" This q-learning algorithm is used for testing and is implemented here: 
    https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb """


def q_learning_for_test(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, env.action_space.n, epsilon)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        action_list = [a for a in range(env.action_space.n)]
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

    return Q, stats


if __name__ == '__main__':
    env = WindyGridworldEnv()
    solution = TemporalDifference(env)

    # Sarsa
    # Q, stats = solution.sarsa(200)
    # plotting.plot_episode_stats(stats)

    # Q-learning
    # Q, stats = solution.q_learning(500)
    # plotting.plot_episode_stats(stats)

    # verify: use q learning implemented by others
    Q, stats = q_learning_for_test(env, 500)
    plotting.plot_episode_stats(stats)
