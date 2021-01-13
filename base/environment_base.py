class FiniteEnvironmentBase:
    """
    Environment in an array
    """

    def __init__(self, num_of_states, transition_matrix=None):
        self.nS = num_of_states
        self.transition_matrix = transition_matrix

    def get_all(self):
        """
        :return: all states
        """
        raise NotImplementedError()

    def step(self, current_state, action):
        """
        Return result of one step look ahead reward
        :param current_state:
        :param action:
        :return: a 2D array if the dynamics are unknown with only one row. Each row contains
            [probability, ]
        """
        raise NotImplementedError()
