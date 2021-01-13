"""
Implementation of linear action value function approximator
"""

import itertools
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

from lib.utils import EpisodeStats, make_epsilon_greedy_policy_with_action_value_estimator


class LinearActionValueFunctionEstimator:
    """ Credit given to:
        https://github.com/dennybritz/reinforcement-learning/blob/master/FA"""
    def __init__(self, env):
        """
        :param env: the environment to be approximated
        """
        self.env = env
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurized representation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def featurize_state(self, state):
        """
        :param state: to be featureized
        :return: featureized state
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        :param s: state to make a prediction for
        :param a:(Optional) action to make a prediction for
        :return:
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
        """
        featurized_state = [self.featurize_state(s)]
        if a is None:  # return prediction across all actions
            predictions = np.zeros(self.env.action_space.n)
            for act in range(self.env.action_space.n):
                predictions[act] = self.models[act].predict(featurized_state)[0]

            return predictions
        else:  # return the prediction with respect to the input action
            return self.models[a].predict(featurized_state)[0]

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        featurized_state = [self.featurize_state(s)]
        self.models[a].partial_fit(featurized_state, [y])


def q_learning_with_linear_func_estimator(env, estimator, episode, gamma=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    :param env: gym environment
    :param estimator: an action-Value function estimator
    :param episode: number of episodes
    :param gamma: discounting factor
    :param epsilon: for epsilon-greedy policy
    :param epsilon_decay: epsilon decaying rate
    :return: an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    stat = EpisodeStats(
        episode_lengths=np.zeros(episode),
        episode_rewards=np.zeros(episode))

    action_list = [a for a in range(env.action_space.n)]

    for i in range(episode):
        policy = make_epsilon_greedy_policy_with_action_value_estimator(
            estimator, env.action_space.n, epsilon * epsilon_decay**i)

        obs = env.reset()

        for t in itertools.count():
            action_prob = policy(obs)
            action = np.random.choice(action_list, p=action_prob)
            next_obs, reward, done, _ = env.step(action)

            stat.episode_lengths[i] = t
            stat.episode_rewards[i] += reward

            q_values_next = estimator.predict(next_obs)
            y = reward + gamma * np.max(q_values_next)
            estimator.update(obs, action, y)

            if done:
                break

            obs = next_obs

    return stat
