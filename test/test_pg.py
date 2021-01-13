from envs.cliff_walking import CliffWalkingEnv
from lib import plotting
from pg import PolicyEstimator, ValueEstimator, actor_critic

if __name__ == '__main__':
    env = CliffWalkingEnv()

    # reinforce with MC
    # policy_estimator = PolicyEstimator(env)
    # value_estimator = ValueEstimator(env)
    #
    # stat = reinforce_with_baseline(env, policy_estimator, value_estimator, 2000,
    #                                gamma=1.0, policy_lr=0.001, value_lr=0.1)
    #
    # plotting.plot_episode_stats(stat, smoothing_window=25)

    # actor critic
    policy_estimator = PolicyEstimator(env)
    value_estimator = ValueEstimator(env)

    stat = actor_critic(env, policy_estimator, value_estimator, 1000,
                        gamma=1.0, policy_lr=0.01, value_lr=0.1)

    plotting.plot_episode_stats(stat, smoothing_window=25)
