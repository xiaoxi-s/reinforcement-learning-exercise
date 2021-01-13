"""
Implementation of Monte Carlo method for tabular cases
"""

import numpy as np

from collections import defaultdict
from lib.utils import make_greedy_policy, make_random_policy, make_epsilon_greedy_policy


class MonteCarlo:
    def __init__(self, env, policy, v=None, q=None):
        self.env = env
        self.policy = policy
        if v is None:
            self.v = defaultdict(float)
        else:
            self.v = v

        if q is None:
            self.q = defaultdict(lambda: np.zeros(env.action_space.n))
        else:
            self.q = q

    def prediction(self, episode, gamma=1.0, mode='e'):
        if mode in 'every':
            return self._every_visit(episode, gamma)
        elif mode in 'first':
            return self._first_visit(episode, gamma)
        else:
            raise TypeError('Mode must be \"first\" or \"every\"')

    def _every_visit(self, episode, gamma):
        # every visit
        counter = defaultdict(int)
        for i in range(1, episode + 1):
            obs = self.env.reset()  # one observation
            g = list()
            observations = []

            while True:
                observations.append(obs)
                obs, reward, done, _ = self.env.step(self.policy(obs))
                g.append(reward)
                if done:
                    break

            # assign the GT to the value of the terminal state
            counter[observations[-1]] = counter[observations[-1]] + 1
            obs = observations[-1]
            self.v[obs] = self.v[obs] + 1 / counter[obs] * (g[-1] - self.v[obs])

            # assign Gt to the value of other states
            j = len(g) - 2
            while j >= 0:
                # calculate G1, G2, ..., GT by dynamic programming
                g[j] = g[j] + gamma * g[j + 1]
                obs = observations[j]
                counter[obs] = counter[obs] + 1
                self.v[obs] = self.v[obs] + 1 / counter[obs] * (g[j] - self.v[obs])
                j -= 1

        return self.v

    def _first_visit(self, episode, gamma):
        # first visit
        returns = defaultdict(list)  # maps from observations to returns
        for i in range(1, episode + 1):
            obs = self.env.reset()  # one observation
            g = list()
            observations = []
            while True:
                observations.append(obs)
                obs, reward, done, _ = self.env.step(self.policy(obs))
                g.append(reward)

                if done:
                    break

            # assign the GT to the value of the terminal state
            # assign Gt to the value of other states
            j = len(g) - 2
            while j >= 0:
                # calculate G1, G2, ..., GT by dynamic programming
                g[j] = g[j] + gamma * g[j + 1]
                j -= 1

            all_obs = set(observations)
            for obs in all_obs:
                ind = observations.index(obs)
                returns[observations[ind]].append(g[ind])

        for obs in returns.keys():
            average_return = sum(returns[obs]) / len(returns[obs])
            self.v[obs] = average_return

        return self.v

    def control_with_epsilon_greedy(self, episode, gamma=1.0, epsilon=0.1):
        policy = make_epsilon_greedy_policy(self.q,  self.env.action_space.n, epsilon)

        counter = defaultdict(int)
        for i in range(1, episode + 1):
            obs = self.env.reset()  # one observation
            g = list()
            observation_action_pairs = []

            while True:
                action = policy(obs)
                observation_action_pairs.append((obs, action))
                obs, reward, done, _ = self.env.step(action)

                g.append(reward)
                if done:
                    break

            # assign the GT to the value of the terminal state
            counter[observation_action_pairs[-1]] = counter[observation_action_pairs[-1]] + 1
            obs_acts = observation_action_pairs[-1]
            obs, action = obs_acts
            self.q[obs][action] = self.q[obs][action] + 1 / counter[obs_acts] * (g[-1] - self.q[obs][action])

            # assign Gt to the value of other states
            j = len(g) - 2
            while j >= 0:
                # calculate G1, G2, ..., GT by dynamic programming
                g[j] = g[j] + gamma * g[j + 1]
                obs_acts = observation_action_pairs[j]
                obs, action = obs_acts
                counter[obs_acts] = counter[obs_acts] + 1
                self.q[obs][action] = self.q[obs][action] + 1 / counter[obs_acts] * (g[j] - self.q[obs][action])
                j -= 1

        return self.q, policy

    def control_ordinary_importance_sampling(self, episode, behavior_policy=None, gamma=1.0):
        """
        Every-visit ordinary importance sampling of MC. Biased estimate of q, although the bias tends to 0 as experience
        increases. Variance can be unbounded The behavior policy is random if behavior_policy is None.
        :param episode: num of episodes
        :param behavior_policy: the behavior policy
        :param gamma: discounting factor
        :return: Q value, and the target (optimal) policy
        """

        def ratio(target_action_probabilities, behavior_action_probabilities, action):
            return target_action_probabilities[action] / behavior_action_probabilities[action]

        if behavior_policy is None:
            behavior_policy = make_random_policy(self.env.nA)

        target_policy = make_greedy_policy(self.q, self.env.nA)

        counter = defaultdict(int)
        action_list = [a for a in range(0, self.env.nA)]

        for i in range(1, episode + 1):
            obs = self.env.reset()  # one observation
            g = list()
            observation_action_pairs = []

            """ Sampling a batch of experience in one episode """
            while True:
                behavior_action_prob = behavior_policy(obs)
                behavior_action = np.random.choice(action_list, p=behavior_action_prob)
                observation_action_pairs.append((obs, behavior_action))
                obs, reward, done, _ = self.env.step(behavior_action)

                g.append(reward)
                if done:
                    break

            """ assign the GT to the value of the terminal state """

            # the last term is calculated separately
            obs_acts = observation_action_pairs[-1]
            obs, behavior_action = obs_acts
            counter[obs_acts] = counter[obs_acts] + 1
            target_action_prob = target_policy(obs)
            product = ratio(target_action_prob, behavior_action_prob, behavior_action)
            self.q[obs][behavior_action] = self.q[obs][behavior_action] + 1 / counter[obs_acts] * \
                                           (product*g[-1] - self.q[obs][behavior_action])

            # calculate gt
            j = len(g) - 2
            while j >= 0:
                # calculate G1, G2, ..., GT by dynamic programming
                g[j] = g[j] + gamma * g[j + 1]

                # standard every visit MC
                obs_acts = observation_action_pairs[j]
                obs, behavior_action = obs_acts
                counter[obs_acts] = counter[obs_acts] + 1

                # the probability ratio
                target_action_prob = target_policy(obs)
                product = product * ratio(target_action_prob, behavior_action_prob, behavior_action)

                # update q value
                self.q[obs][behavior_action] = self.q[obs][behavior_action] + 1 / counter[obs_acts] * \
                                               (product * g[j] - self.q[obs][behavior_action])
                j -= 1

        return self.q, target_policy

    def control_weighted_importance_sampling(self, episode, behavior_policy=None, gamma=1.0):
        """
        Every-visit ordinary importance sampling of MC. Biased estimate of q, although the bias tends to 0 as experience
        increases. Variance is dramatically lower. The behavior policy is random if behavior_policy is None.
        :param episode: num of episode
        :param behavior_policy: the behavior policy
        :param gamma: discounting factor
        :return: q and the policy function
        """

        C = defaultdict(lambda: np.zeros(self.env.action_space.n))  # cumulated sum of the denominator

        if behavior_policy is None:
            behavior_policy = make_random_policy(self.env.nA)
        target_policy = make_greedy_policy(self.q, self.env.nA)
        action_list = [a for a in range(0, self.env.nA)]

        for i in range(1, episode+1):
            obs = self.env.reset()  # one observation
            g = list()
            observation_action_pairs = []

            """ Sampling a batch of experience in one episode """
            while True:
                behavior_action_prob = behavior_policy(obs)
                behavior_action = np.random.choice(action_list, p=behavior_action_prob)
                observation_action_pairs.append((obs, behavior_action))
                obs, reward, done, _ = self.env.step(behavior_action)

                g.append(reward)
                if done:
                    break

            G = 0
            W = 1.0

            for t in range(0, len(observation_action_pairs))[::-1]:
                obs, action = observation_action_pairs[t]
                G = g[t] + gamma * G
                C[obs][action] += W
                self.q[obs][action] = self.q[obs][action] + (W/C[obs][action]) * (G - self.q[obs][action])

                # find one action not taken by the target policy
                if action != np.argmax(target_policy(obs)):
                    break

                W = W*1./behavior_policy(obs)[action]

        return self.q, target_policy

