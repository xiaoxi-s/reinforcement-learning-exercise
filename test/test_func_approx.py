import gym
import matplotlib

from func_approx import LinearActionValueFunctionEstimator, q_learning_with_linear_func_estimator
from lib import plotting

matplotlib.style.use('ggplot')

if __name__ == '__main__':
    env = gym.envs.make("MountainCar-v0")

    estimator = LinearActionValueFunctionEstimator(env)
    stat = q_learning_with_linear_func_estimator(env, estimator, 2000, epsilon=0.0)

    plotting.plot_cost_to_go_mountain_car(env, estimator)
    plotting.plot_episode_stats(stat, smoothing_window=25)
