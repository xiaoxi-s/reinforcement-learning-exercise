import numpy as np

from base.environment_base import FiniteEnvironmentBase
from base.policy_base import FinitePolicyBase


class GridWorld(FiniteEnvironmentBase):
    """
    States:
        0  1  2  3
        4  5  6  7
        8  9  10 11
        12 13 14 15
    """

    def __init__(self):
        super().__init__(num_of_states=16)
        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    def get_all(self):
        """
        :return: all states
        """
        return np.array(self.states, dtype=np.int32)

    def step(self, current_state, action):
        """
        One step look ahead
        :param current_state:
        :param action:
        :return: 2D array with only one row because of the deterministic policy and env dynamics
            each row contains (prob, next_state, reward, done)
        """

        assert current_state in self.states
        assert action in [0, 1, 2, 3]

        # terminal state
        if current_state == 0 or current_state == 15:
            return [[1, current_state, 0, True]]

        # boundaries
        if current_state in [1, 2, 3] and action == 0:
            return [[1, current_state, -1, False]]
        if current_state in [12, 13, 14] and action == 1:
            return [[1, current_state, -1, False]]
        if current_state in [4, 8, 12] and action == 2:
            return [[1, current_state, -1, False]]
        if current_state in [3, 7, 11] and action == 3:
            return [[1, current_state, -1, False]]

        # determine offset
        if action == 0:
            offset = -4
        elif action == 1:
            offset = 4
        elif action == 2:
            offset = -1
        else:
            offset = 1

        next_state = current_state + offset
        return [[1, next_state, -1, True if next_state == 0 else False]]


class GridWorldPolicy(FinitePolicyBase):
    """
    Policy:
        move up    - 0
        move down  - 1
        move left  - 2
        move right - 3
    """

    def __init__(self):
        super().__init__(num_of_actions=4)
        self.actions = [0, 1, 2, 3]

        # 16 states, 4 actions
        self.action_probabilities = np.ones((16, 4)) / 4

    def get_all(self):
        """
        :return: all actions
        """
        return np.array(self.actions, dtype=np.int32)

    def act_prob(self, state, action):
        """
        :param state:
        :param action:
        :return: the probability of taking an action in a state
        """
        return self.action_probabilities[state][action]

    def act(self, state):
        """
        :param state:
        :return: the probabilities of taking each action in a state
        """
        return self.action_probabilities[state]

    def update(self, q, state):
        """
        Update value function given q(s, .)
        :param q: action value function
        :param state: the given state
        :return: None
        """
        action_index = np.argmax(q[state])
        updated_prob = np.zeros((4,))
        updated_prob[action_index] = 1
        self.action_probabilities[state] = updated_prob
