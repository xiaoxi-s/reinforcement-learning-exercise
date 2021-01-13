class FinitePolicyBase:
    """
    Policy in an array
    """

    def __init__(self, num_of_actions, deterministic=True):
        self.nA = num_of_actions

    def get_all(self):
        """
        :return: all actions
        """
        raise NotImplementedError()

    def act_prob(self, state, action):
        """
        :param state:
        :param action:
        :return: the probability of taking an action in a state
        """
        raise NotImplementedError()

    def act(self, state):
        """
        :param state: the state where the agent would take an action
        :return: the probabilities of taking each action in a state
        """
        raise NotImplementedError()

    def update(self, q, state):
        """
        Update policy given action value function in a state
        :param q: action value function
        :param state: the given state
        :return: None
        """
        raise NotImplementedError()
