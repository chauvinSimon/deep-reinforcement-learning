import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# DDPG = deterministic deep policy gradient
# https://arxiv.org/pdf/1509.02971.pdf

# In original paper, low-dimensional networks had 2 hidden layers
# with 400 and 300 units respectively (≈ 130,000 parameters)


def hidden_init(layer):
    """
    compute the bounds [-lim, lim] for subsequent uniform sampling
    with lim = 1/sqrt(nb_output)
    """
    # non-terminal layers are initialized from uniform distributions, [− 1/√f , 1/√f ]
    # where f is the fan-in of the layer
    # fan-in = the number of inputs to a hidden unit
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor (Policy) Model.
    The actor estimates optimal policy deterministically – giving the best action
    - not a distribution over the action
    - here the actor learns argmax[action](Q{s,a}) = best action
    """

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn3 = nn.BatchNorm1d(action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        # to get each action in [-1, 1]
        return torch.tanh(x)


class Critic(nn.Module):
    """ Critic (Value) Model
    The critic learns to evaluate the optimal Q-function
    - Uses the actor’s best believed action
    - This gives the TD-target
    - Like argmax of DQN """

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Actions are not included until the 2nd hidden layer of Q
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

        self.reset_parameters()

    def reset_parameters(self):
        # as in the original paper, the final layer weights and biases of both the actor and critic
        # are initialized from a uniform distribution [−3×10−3, 3×10−3]
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state = self.bn0(state)
        xs = self.fc1(state)
        xs = self.bn1(xs)
        xs = F.leaky_relu(xs)
        # we want to assess an (state, action) pair
        x = torch.cat((xs, action), dim=1)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        return self.fc3(x)
