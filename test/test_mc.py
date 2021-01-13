from collections import defaultdict

import numpy as np

from envs.blackjack import BlackjackEnv
from lib import plotting
from mc import MonteCarlo


def naive_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


if __name__ == '__main__':
    env = BlackjackEnv()

    # every visit
    # solution = MonteCarlo(env, naive_policy)
    #
    # V_10k = solution.prediction(episode=10000, gamma=1, mode='e')
    # plotting.plot_value_function(V_10k, title="10,000 Steps")
    #
    # solution = MonteCarlo(env, naive_policy)
    # V_500k = solution.prediction(episode=500000, gamma=1, mode='e')
    # plotting.plot_value_function(V_500k, title="500,000 Steps")

    # on-policy control with epsilon greedy
    # solution = MonteCarlo(env, naive_policy)
    # Q, policy = solution.control_with_epsilon_greedy(episode=500000, gamma=1, epsilon=0.1)
    #
    # V = defaultdict(float)
    # for state, actions in Q.items():
    #     action_value = np.max(actions)
    #     V[state] = action_value
    # plotting.plot_value_function(V, title="Optimal Value Function")

    # off-policy
    # ordinary importance sampling: behavior policy is uniformly random
    # solution = MonteCarlo(env, naive_policy)
    # Q, policy = solution.control_ordinary_importance_sampling(episode=500000)
    # V = defaultdict(float)
    # for state, action_values in Q.items():
    #     action_value = np.max(action_values)
    #     V[state] = action_value
    # plotting.plot_value_function(V, title="Optimal Value Function")

    # weighted importance sampling: behavior policy is uniformly random
    solution = MonteCarlo(env, naive_policy)
    Q, policy = solution.control_weighted_importance_sampling(episode=500000)
    V = defaultdict(float)
    for state, action_values in Q.items():
        action_value = np.max(action_values)
        V[state] = action_value
    plotting.plot_value_function(V, title="Optimal Value Function")
