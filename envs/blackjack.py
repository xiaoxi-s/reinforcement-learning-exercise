"""
Credit given to:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/blackjack.py
"""

import gym

from gym import spaces
from gym.utils import seeding

# ace, num (2-10), king (last 3 tens)
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def get_reward(player_hand, dealer_hand):
    if score(player_hand) == score(dealer_hand):
        # draw
        return 0
    elif score(player_hand) > score(dealer_hand):
        # win
        return 1
    else:
        # lose
        return -1


def draw_card(np_random):
    return np_random.choice(deck)


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):
    return sum_hand(hand) > 21


def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21


def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):
    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # players current sum
            spaces.Discrete(11),  # one card shown by the dealer (1-10)
            spaces.Discrete(2)  # whether the player holds an ace ()
        ))
        self.np_random = None  # random seed
        self.player = None  # cards in the player's hand
        self.dealer = None  # cards in the dealer's hand

        self.seed()
        self.natural = natural
        self.nA = 2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # action is hit
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # action is stick
            done = True
            # dealer would keep drawing until sum exceeds 17
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = get_reward(self.player, self.dealer)
            # if natural is allowed and the player has a natural
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5

        # (observation, reward, done, info) combination
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player)), reward, done, None

    def reset(self):
        self.player = draw_hand(self.np_random)
        self.dealer = draw_hand(self.np_random)

        # ensure that the player's score is no less than 12 once the game is started
        while sum_hand(self.player) < 12:
            self.player.append(draw_card(self.np_random))

        return sum_hand(self.player), self.dealer[0], usable_ace(self.player)
