"""
Implementation of Dynamic Programming for planning
"""

import numpy as np

from base.policy_base import FinitePolicyBase
from base.environment_base import FiniteEnvironmentBase


class DynamicProgramming:
    def __init__(self, env, policy, v=None):
        """
        Init necessary variables
        :param env: the environment
        :param policy: the policy
        :param v: initial value function
        """
        # finite policy and environment base
        if not isinstance(policy, FinitePolicyBase) or not isinstance(env, FiniteEnvironmentBase):
            raise TypeError('Either policy or environment is not finite')

        self.env = env
        self.policy = policy

        self.states = env.get_all()
        self.actions = policy.get_all()
        self.nS = env.nS
        self.nA = policy.nA

        if np.isinf(self.nS) or np.isinf(self.nA):
            raise TypeError("State or action space is infinite")

        if v is not None:
            self.v = v
        else:
            self.v = dict()

    def policy_evaluation(self, gamma=1.0, epsilon=1e-6):
        """
        Evaluate a policy with dynamic programming
        :param gamma: the discounting factor
        :param epsilon: threshold for error between two consecutive updates
        :return: the value function
        """
        while True:
            delta = 0
            # for each state that V[s] would be updated
            for s in self.states:
                temp_v = 0
                # for each policy that could be taken
                for a, action_prob in enumerate(self.policy.act(s)):
                    # known outcomes of each step
                    for prob, next_state, reward, done in self.env.step(s, a):
                        temp_v += action_prob * prob * (reward + gamma * self.v[next_state])

                delta = max(delta, abs(self.v[s] - temp_v))
                self.v[s] = temp_v

            if delta < epsilon:
                break
        return self.v

    def policy_iteration(self, gamma=1.0, epsilon=1e-6):
        """
        Policy iteration algorithm
        :param gamma: discounting factor
        :param epsilon: error that is tolerable
        :return: best policy with epsilon error resulted from value function
        """
        q = np.zeros((self.nS, self.nA))
        policy_stable = False  # must improve once

        while True:
            self.policy_evaluation(gamma, epsilon)
            for s in self.states:
                for a in self.actions:
                    temp_q = 0
                    for prob, next_state, reward, done in self.env.step(s, a):
                        temp_q += prob * (reward + gamma * self.v[next_state])
                    q[s][a] = temp_q

                # policy.act returns the probability of taking each action
                current_action = np.argmax(self.policy.act(s))
                improved_action = np.argmax(q[s])

                # ensures first time update to break ties because of initialization
                if policy_stable is False or current_action != improved_action:
                    policy_stable = False
                    self.policy.update(q, s)

            if policy_stable:
                break
            policy_stable = True
        return self.policy

    def value_iteration(self, gamma=1.0, epsilon=1e-6):
        while True:
            delta = 0
            for s in self.states:
                # store action values in a given state in an array
                temp_action_values = np.zeros((len(self.actions),))
                for i, a in enumerate(self.actions):
                    for prob, next_state, reward, done in self.env.step(s, a):
                        temp_action_values[i] += prob * (reward + gamma * self.v[next_state])

                max_action_value = np.max(temp_action_values)
                delta = max(delta, abs(max_action_value - self.v[s]))
                self.v[s] = max_action_value

            if delta < epsilon:
                break

        # calculate action values and update the policy greedily
        q = np.zeros((self.nS, self.nA))
        for s in self.states:
            for a in self.actions:
                for prob, next_state, reward, done in self.env.step(s, a):
                    q[s][a] += prob * (reward + gamma * self.v[next_state])
            self.policy.update(q, s)

        return self.policy

    @classmethod
    def policy_evaluation_in_matrix_form(cls, transition_matrix, rewards, v=None, gamma=0.9, epsilon=1e-6):
        """
        If the transition matrix, immediate rewards are known
        :param transition_matrix: the transition probability matrix (nS x nS) given a policy
        :param rewards: the immediate reward vector (nS x 1)
        :param v: a vector where each entry denotes the value of each state (nS x 1)
        :param gamma: the discounting factor
        :param epsilon: threshold for error between two consecutive updates
        :return: the value function
        """
        if v is None:
            v = np.zeros(transition_matrix.shape[0])

        while True:
            temp_v = rewards + gamma * v
            delta = np.max(np.abs(temp_v - v))
            v = temp_v

            if delta < epsilon:
                break
        return v
