# Definition of the Agent, the Memory Replay and the OU-Noise
# DDPG = deterministic deep policy gradient
# https://arxiv.org/pdf/1509.02971.pdf
# so code walk-through: https://www.youtube.com/watch?v=08V9r3NgFSE

###
# Main features:
# -1- Experience Replay buffer
#    critic network is trained off-policy with samples from a replay buffer to minimize correlations between samples
# -2- Target nets to give consistent targets
# 	 using soft target updates
# -3- Batch Normalization to minimize covariance shift during training
#    by ensuring that each layer receives whitened input
#    (especially useful when trying to generalize over different environments)
# -4- OU-Noise
#    to construct an exploration policy µ by adding noise sampled from a noise process N to our actor policy
#    i.e. generate temporally correlated exploration for exploration efficiency with inertia
###

import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle

BUFFER_SIZE = int(5e5)  # replay buffer size (int(1e6) in paper)
BATCH_SIZE = 512        # mini-batch size (64 in paper)
GAMMA = 0.99            # discount factor (0.99 in paper)
TAU = 1e-3              # for soft update of target parameters (1e-3 in paper)
LR_ACTOR = 1e-3         # learning rate of the actor (1e-4 in paper)
LR_CRITIC = 3e-3        # learning rate of the critic (1e-3 in paper)
WEIGHT_DECAY = 0        # L2 weight decay (1e−2 in paper)
NUM_AGENTS = 20
# OPTIMIZER = Adam      # (as in paper)
# ACTIVATION = ReLu     # for hidden layers (as in paper)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment
    DDQN is inspired from DQN: the network is trained with a target Q network
        to give consistent targets during temporal-difference backups
    """

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # As in paper, initial hard copy
        self.soft_update(self.critic_local, self.critic_target, tau=0.99)
        self.soft_update(self.actor_local, self.actor_target, tau=0.99)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        self.learning_counter = 0
        self.step_counter = 0

        # debug - monitoring tools of action distribution
        self.stds = []
        self.means = []

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn
        Notes:
            - the original paper learns at each step
            - here, get less aggressive (update networks 10 times after every 20 timesteps
        """
        self.step_counter += 1
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # learn, if enough samples are available in memory, 10 times, every 20 steps
        if len(self.memory) > BATCH_SIZE and self.step_counter % (20*NUM_AGENTS) == 0:
            # print('\rStep {} - time to learn'.format(self.step_counter), end="\n")
            for _ in range(20):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        # Sets the module in *training mode* (as opposed to the evaluation mode)
        self.actor_local.train()
        if add_noise:
            for i in range(NUM_AGENTS):
                action[i] += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        """ Initialize a random process N for action exploration """
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # debug: monitor the distribution in actions
        self.learning_counter += 1
        if self.learning_counter % 1000 == 0:
            self.means.append(actions.cpu().data.numpy().mean(0))
            self.stds.append(actions.cpu().data.numpy().std(0))
            logfile_name = "means.txt"
            with open(logfile_name, "wb") as fp:
                pickle.dump(self.means, fp)
            logfile_name = "stds.txt"
            with open(logfile_name, "wb") as fp:
                pickle.dump(self.stds, fp)

            info = "\nactions in batch at {}-th learning:\n\t shape = {},\n\t mean = {},\n\t  std = {}".format(
                self.learning_counter, np.shape(actions.cpu().data.numpy()), actions.cpu().data.numpy().mean(0),
                actions.cpu().data.numpy().std(0))
            print(info)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        # using the next_action to produce a target
        # hence sort of on-policy SARSA. Except that it uses the target net
        q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        # Compute critic loss (L2)
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        # backward() function accumulates gradients - zero them out at the start
        self.critic_optimizer.zero_grad()
        # The backward function receives the gradient of the output Tensors
        #   with respect to some scalar value, and computes the gradient of the
        #   input Tensors with respect to that same scalar value.
        critic_loss.backward()
        # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
        # These are accumulated into x.grad for every parameter x.
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        # optimizer.step is performs a parameter update based on the current gradient
        # (stored in .grad attribute of a parameter) and the update rule.
        self.critic_optimizer.step()
        # For instance, SGD optimizer performs: x += -lr * x.grad

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss - using sampled policy gradient
        actions_predict = self.actor_local(states)
        # use samples to estimate the expectation. Hence mean()
        # Deterministic Gradient Policy Theorem: gradient = expectation[Q-values]
        # pytorch by default does gradient DESCENT. Hence minus term for ASCENT
        actor_loss = -self.critic_local(states, actions_predict).mean()

        # Minimize the loss: Zero gradients, perform a backward pass, and update the weights
        self.actor_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        actor_loss.backward()  # Use autograd to compute the backward pass
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters, rather than directly copying the weights
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """exploration noise based on Ornstein-Uhlenbeck process
    Ideas:
    - In action += self.noise.sample()
    - Construct an exploration policy µ by adding noise sampled from a noise process N to our actor policy
    - Use temporally correlated noise in order to explore well in physical environments that have momentum
    - Original paper sets θ = 0.15 and σ = 0.2.
    - The Ornstein-Uhlenbeck process models the velocity of a Brownian particle with friction,
    which results in temporally correlated values centered around 0.
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.state = None
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        # ToDo: P.E.R.
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples
    DDQN is inspired from DQN: ReplayBuffer to minimize correlations between samples
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # Uniform sampling for the moment. ToDo = P.E.R.
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
