# Definition of one DDPG agent
# See networkforall.py for the nets

# hyper-parameters:
# - [Adam with learning rate of 0.01]o.p.
# - [Adam with learning rate of 10−4 and 10−3 for the actor and critic respectively]ddpg.p.
# - [For Q we included L2 weight decay of 10−2]ddpg.p.
# - "Set L2 weight decay on the critic to 0" -- @danielnbarbosa
#       (positive reward signal is pretty sparse in the beginning
#        and this can easily get drowned out by the weight decay)

from networkforall import Network
from utilities import hard_update
from torch.optim import Adam

# add OU noise for exploration
from OUNoise import OUNoise

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


class DDPGAgent:
    """
    4 nets per agent: actor + critic + target_actor + target_critic
    """
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic,
                 hidden_in_critic, hidden_out_critic, lr_actor=1.0e-4, lr_critic=1.0e-3):
        super(DDPGAgent, self).__init__()

        self.local_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.local_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        self.noise = OUNoise(out_actor, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.local_actor)
        hard_update(self.target_critic, self.local_critic)

        self.actor_optimizer = Adam(self.local_actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.local_critic.parameters(), lr=lr_critic, weight_decay=0)

    def act(self, obs, noise=0.0):
        obs = obs.to(device)  # transfer the observation tensor to device
        self.local_actor.eval()
        with torch.no_grad():
            action = self.local_actor(obs)
        self.local_actor.train()
        action += noise*self.noise.noise().to(device)
        return action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        self.target_actor.eval()
        with torch.no_grad():
            action = self.target_actor(obs)
        self.target_actor.train()
        action += noise*self.noise.noise().to(device)
        return action

    def reset(self):
        self.noise.reset()

