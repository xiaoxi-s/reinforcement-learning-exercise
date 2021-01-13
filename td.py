"""
Implementation of Temporal Difference learning for tabular cases
"""

import itertools
import numpy as np

from collections import defaultdict
from lib.utils import make_epsilon_greedy_policy, make_greedy_policy, EpisodeStats


class TemporalDifference:
    def __init__(self, env, q=None):
        self.env = env

        if q is None:
            self.q = defaultdict(lambda: np.zeros(env.action_space.n))
        else:
            self.q = q

    def sarsa(self, episode, gamma=1.0, alpha=0.5, epsilon=0.1):
        """
        Sarsa algorithm: On-policy control. Finds the optimal epsilon-greedy policy.

        :param episode:Number of episodes to run for.
        :param gamma: discounting factor
        :param alpha: TD learning rate
        :param epsilon: for epsilon-greedy policy
        :return: action value, episode_rewards (2D array where each row is one episode of rewards)
        """

        policy = make_epsilon_greedy_policy(self.q, self.env.nA, epsilon)

        stats = EpisodeStats(
            episode_lengths=np.zeros(episode),
            episode_rewards=np.zeros(episode))

        for i in range(0, episode):
            obs = self.env.reset()
            action = policy(obs)

            for t in itertools.count():
                next_obs, reward, done, _ = self.env.step(action)
                next_action = policy(next_obs)
                self.q[obs][action] = self.q[obs][action] + \
                                      alpha*(reward + gamma*self.q[next_obs][next_action] - self.q[obs][action])

                stats.episode_lengths[i] = t
                stats.episode_rewards[i] += reward

                if done:
                    break

                obs = next_obs
                action = next_action

        return self.q, stats

    def q_learning(self, episode, gamma=1.0, alpha=0.5, epsilon=0.1):
        """
        Q-Learning algorithm: Off-policy TD control.
            Behavior policy: an epsilon-greedy policy
            Target policy: an optimal policy

        :param episode: Number of episodes to run for.
        :param gamma: discounting factor
        :param alpha: TD learning rate
        :param epsilon: for epsilon-greedy policy
        :return: action value, statistics
        """

        behavior_policy = make_epsilon_greedy_policy(self.q, self.env.nA, epsilon)
        target_policy = make_greedy_policy(self.q, self.env.nA)

        stats = EpisodeStats(
            episode_lengths=np.zeros(episode),
            episode_rewards=np.zeros(episode))

        action_list = [a for a in range(self.env.nA)]

        for i in range(0, episode):
            obs = self.env.reset()

            for t in itertools.count():
                # behave
                action = behavior_policy(obs)
                next_obs, reward, done, _ = self.env.step(action)

                # target
                next_action_prob = target_policy(next_obs)
                next_action = np.random.choice(action_list, p=next_action_prob)

                # update rule
                self.q[obs][action] = self.q[obs][action] + \
                                      alpha*(reward + gamma*self.q[next_obs][next_action] - self.q[obs][action])

                stats.episode_lengths[i] = t
                stats.episode_rewards[i] += reward

                if done:
                    break

                obs = next_obs

        return self.q, stats
