# OU-Noise
#    to construct an exploration policy µ by adding noise sampled from a noise process N to our actor policy
#    i.e. generate temporally correlated exploration for exploration efficiency with inertia

# hyper-parameters
# - [We used an Ornstein-Uhlenbeck process with θ = 0.15 and σ = 0.2]ddpg.p.

import numpy as np
import torch


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    """exploration noise based on Ornstein-Uhlenbeck process
    Ideas:
    - In action += self.noise.sample()
    - Construct an exploration policy µ by adding noise sampled from a noise process N to our actor policy
    - Use temporally correlated noise in order to explore well in physical environments that have momentum
    - Original DDQN paper sets θ = 0.15 and σ = 0.2.
    - The Ornstein-Uhlenbeck process models the velocity of a Brownian particle with friction,
    which results in temporally correlated values centered around 0.
    """

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()
