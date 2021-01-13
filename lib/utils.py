from collections import namedtuple

import numpy as np

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def make_epsilon_greedy_policy(q, nA, epsilon):
    """
    :param q: the action-value function
    :param epsilon: probability of choosing an action randomly
    :param nA: num of actions
    :return: an epsilon greedy policy
    """

    def epsilon_greedy_policy(obs):
        chance = np.random.uniform(0, 1)
        if chance < epsilon:
            # pick action randomly
            action = np.random.choice(range(0, nA))
        else:
            # pick the best action based on q
            action = np.argmax(q[obs])

        return action

    return epsilon_greedy_policy


def make_epsilon_greedy_policy_with_action_value_estimator(estimator, nA, epsilon):
    """
    :param estimator: action-value function estimator
    :param epsilon: probability of choosing an action randomly
    :param nA: num of actions
    :return: A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def epsilon_greedy_policy_with_estimator(obs):
        action_probabilities = np.ones(nA, dtype=float) * epsilon / nA
        y = estimator.predict(obs)
        best_action = np.argmax(y)
        action_probabilities[best_action] += 1 - epsilon

        return action_probabilities

    return epsilon_greedy_policy_with_estimator


def make_random_policy(nA):
    """
    Credit to:
        https://github.com/dennybritz/reinforcement-learning/tree/master/MC
    :return: a random policy
    """
    action_probabilities = np.ones(nA, dtype=float) / nA

    def random_policy(obs):
        """
        :param obs: an observation
        :return: a vector of action probability
        """
        return action_probabilities

    return random_policy


def make_greedy_policy(q, nA):
    """
    Create the greedy policy with respect to Q
    :param q: the action value function
    :param nA: number of actions
    :return: a greedy policy with respect to the action value function
    """

    def policy(obs):
        """
        :param obs: an observation
        :return: a vector of action probability
        """
        action_probabilities = np.zeros(nA, dtype=float)
        action = np.argmax(q[obs])
        action_probabilities[action] = 1.0

        return action_probabilities

    return policy
