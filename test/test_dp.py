import numpy as np

from dp import DynamicProgramming
from envs.gridworld import GridWorld, GridWorldPolicy

if __name__ == '__main__':
    world = GridWorld()
    policy = GridWorldPolicy()

    # initialize value function
    v = np.zeros((16,), dtype=np.float64)

    solution = DynamicProgramming(world, policy, v)
    v = np.resize(solution.policy_evaluation(1.0, 1e-6), (4, 4))
    print(v)
    p = solution.policy_iteration(1, 1e-6)
    print(p.action_probabilities)
    v = np.resize(solution.policy_evaluation(1.0, 1e-6), (4, 4))
    print(v)

    # value iteration
    world = GridWorld()
    policy = GridWorldPolicy()
    # initialize value function
    v = np.zeros((16,), dtype=np.float64)
    solution = DynamicProgramming(world, policy, v)
    p = solution.value_iteration(1.0, 1e-6)
    print(p.action_probabilities)
