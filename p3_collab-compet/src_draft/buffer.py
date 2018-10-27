# Experience Replay buffer
# - critic network is trained off-policy with samples from a replay buffer to minimize correlations between samples

from collections import deque
import random
from utilities import transpose_list


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples
    DDPG is inspired from DQN: ReplayBuffer to minimize correlations between samples
    """

    def __init__(self, size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self, transition):
        """push into the buffer"""

        input_to_buffer = transpose_list(transition)

        for item in input_to_buffer:
            self.deque.append(item)

    def sample(self, batch_size):
        """sample from the buffer"""
        samples = random.sample(self.deque, batch_size)

        # transpose list of list
        return transpose_list(samples)

    def __len__(self):
        """current size of the buffer"""
        return len(self.deque)
