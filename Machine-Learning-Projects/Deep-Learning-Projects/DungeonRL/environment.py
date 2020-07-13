import random
from enums import *


class DungeonSimulator:
    def __init__(self, length=5, slip=0.1, small=2, large=10):
        self.length = length    # The length of the dungeon
        self.slip = slip        # Probability of slipping(reversing) an action
        self.small = small      # Reward for BACKWARD action
        self.large = large      # Reward for reaching the end of the dungeon
        self.state = 0          # The position of the agent in the dungeon

    def take_action(self, action):
        if random.random() < self.slip:
            action = not action     # Reverse the action
        if action == BACKWARD:
            reward = self.small
            self.state = 0
        if action == FORWARD:
            if self.state < self.length - 1:
                self.state += 1
                reward = 0
            else:
                reward = self.large

        return self.state, reward

    def reset(self):
        self.state = 0
        return self.state
