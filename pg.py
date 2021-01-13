"""
Implementation of policy gradient methods. Estimators are linear.
"""

import torch
import itertools
import numpy as np
from lib.utils import EpisodeStats


def to_one_hot(index, num_of_classes):
    one_hot = np.zeros((num_of_classes,))
    one_hot[index] = 1
    one_hot = torch.from_numpy(one_hot)
    one_hot.double()

    return one_hot


class PolicyEstimator(torch.nn.Module):
    """
    Linear policy estimator
    """

    def __init__(self, env, weight_initializer=torch.nn.init.zeros_):
        super().__init__()
        self.input_layer = torch.nn.Linear(int(env.observation_space.n), int(env.action_space.n))
        self.output_layer = torch.nn.Softmax(dim=-1)
        weight_initializer(self.input_layer.weight)

    def forward(self, state):
        output = self.input_layer(state)
        output = self.output_layer(output)

        return output


class ValueEstimator(torch.nn.Module):
    """
    Linear state value estimator
    """

    def __init__(self, env, weight_initializer=torch.nn.init.zeros_):
        super().__init__()
        self.input_layer = torch.nn.Linear(env.observation_space.n, 1)
        weight_initializer(self.input_layer.weight)

    def forward(self, state):
        output = self.input_layer(state)

        return output


def reinforce_with_baseline(env, policy_estimator, value_estimator, episode, gamma=1.0,
                            policy_lr=0.01, value_lr=0.1, op=torch.optim.Adam):
    """
    :param env: gym environment
    :param policy_estimator: policy estimator
    :param value_estimator: state value estimator
    :param episode: num of episode
    :param gamma: discounting factor
    :param value_lr: learning rate for value estimator
    :param policy_lr: learning rate for policy estimator
    :param op: optimizer
    :return: an EpisodeState object
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy_estimator.to(device)
    value_estimator.to(device)

    policy_estimator.double()
    value_estimator.double()

    value_optimizer = op(value_estimator.parameters(), lr=value_lr)
    policy_optimizer = op(policy_estimator.parameters(), lr=policy_lr)

    stat = EpisodeStats(
        episode_lengths=np.zeros(episode),
        episode_rewards=np.zeros(episode)
    )
    action_list = [a for a in range(env.action_space.n)]

    for i in range(episode):

        print('Episode {}'.format(i))

        obs_action_pairs = []
        obs = env.reset()

        # generate experience sequence S0, A0, R1, S1, A1, ...
        for t in itertools.count():
            obs_one_hot = to_one_hot(obs, env.observation_space.n)
            action_prob = policy_estimator(obs_one_hot).detach().numpy()
            action = np.random.choice(action_list, p=action_prob)

            next_obs, reward, done, _ = env.step(action)
            obs_action_pairs.append((obs, reward, action))

            stat.episode_lengths[i] = t
            stat.episode_rewards[i] += reward

            if done:
                obs_action_pairs.append([next_obs, 0, action])
                break

            obs = next_obs

        total_return = 0
        # update estimators
        value_optimizer.zero_grad()
        policy_optimizer.zero_grad()

        for j in range(len(obs_action_pairs))[::-1]:
            obs, reward, action = obs_action_pairs[j]

            # true total return
            total_return += reward + gamma * total_return

            # MC estimate
            obs_one_hot = to_one_hot(obs, env.observation_space.n)
            estimated_return = value_estimator(obs_one_hot)

            # update two estimators
            advantage = total_return - estimated_return
            policy_loss = -torch.log(policy_estimator(obs_one_hot).select(0, action)) * float(advantage)
            policy_loss.backward(retain_graph=True)
            policy_optimizer.step()

            value_loss = torch.square(total_return - estimated_return)
            value_loss.backward()
            value_optimizer.step()

        # print(total_return)

    return stat


def actor_critic(env, policy_estimator, value_estimator, episode, gamma=1.0,
                 policy_lr=0.01, value_lr=0.1, op=torch.optim.Adam):
    """
    :param env: gym environment
    :param policy_estimator: policy estimator
    :param value_estimator: state value estimator
    :param episode: num of episode
    :param gamma: discounting factor
    :param value_lr: learning rate for value estimator
    :param policy_lr: learning rate for policy estimator
    :param op: optimizer
    :return: an EpisodeStat object
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy_estimator.to(device)
    value_estimator.to(device)

    policy_estimator.double()
    value_estimator.double()

    value_optimizer = op(value_estimator.parameters(), lr=value_lr)
    policy_optimizer = op(policy_estimator.parameters(), lr=policy_lr)

    stat = EpisodeStats(
        episode_lengths=np.zeros(episode),
        episode_rewards=np.zeros(episode)
    )
    action_list = [a for a in range(env.action_space.n)]

    for i in range(episode):

        print('Episode {}'.format(i))

        obs = env.reset()

        k = 0

        value_optimizer.zero_grad()
        policy_optimizer.zero_grad()

        # generate experience sequence S0, A0, R1, S1, A1, ...
        for t in itertools.count():
            obs_one_hot = to_one_hot(obs, env.observation_space.n)
            action_prob = policy_estimator(obs_one_hot).detach().numpy()
            action = np.random.choice(action_list, p=action_prob)

            # store the next observation
            next_obs, reward, done, _ = env.step(action)

            stat.episode_lengths[i] = t
            stat.episode_rewards[i] += reward

            # update
            obs_one_hot = to_one_hot(obs, env.observation_space.n)
            next_obs_one_hot = to_one_hot(next_obs, env.observation_space.n)

            # calculate TD terms
            td_target = reward + gamma * float(value_estimator(next_obs_one_hot))
            td_error = td_target - value_estimator(obs_one_hot)

            # update two estimators
            policy_loss = -torch.log(policy_estimator(obs_one_hot).select(0, action)) * float(td_error)
            policy_loss.backward()
            policy_optimizer.step()

            value_loss = torch.square(td_error)
            value_loss.backward()
            value_optimizer.step()

            if done:
                # terminal states treated as an infinite loop
                break

            obs = next_obs

            # do not exceed 1000
            k += 1
            if k > 1000:
                break

    return stat


def actor_critic_in_continuing_space(env, policy_estimator, value_estimator, episode, gamma=1.0,
                                     policy_lr=0.01, value_lr=0.1, op=torch.optim.Adam):
    pass


