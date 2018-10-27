# In MADDPG, each agent has its own actor and its own critic
# original [Paper](http://arxiv.org/abs/1706.02275)

# responsible for
# - action taking
# - model updates (policy + critic)
# see ddpg.py for other details of one DDPG agent

# hyper-parameters
# - [policies are parameterized by a 2-layer ReLU MLP with 64 units per layer]o.p.
# - [2 hidden layers with 400 and 300 units respectively]ddpg.p.
# - [actions were not included until the 2nd hidden layer of Q]ddpg.p.

# Features:
#  - For Critic, use Huber-loss (less sensitive to outliers than the squared error loss)
#    - quadratic for small values of [target-estimate], and linear for large values

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


class MADDPG:
    def __init__(self, discount_factor=0.95, tau=0.01):
        super(MADDPG, self).__init__()

        # args = in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic
        # critic input = obs_full + actions = 2*24+2+2=52
        self.maddpg_agent = [DDPGAgent(24, 64, 64, 2, 52, 64, 64),
                             DDPGAgent(24, 64, 64, 2, 52, 64, 64)]
        # DDPGAgent(24, 16, 8, 2, 52, 32, 16)]

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def act(self, obs_all_agents, noise=0.0):
        """get local network actions from all agents"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [agent.target_act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number, logger):
        """
        update critic and actor nets of one given agent
        :param samples: a list, representing the batch and containing 7 elements
        samples = [obs, obs_full, action, reward, next_obs, next_obs_full, done]
        with e.g. done = [[False, False], [False, False], False, False]] for batch_size = 3
        :param agent_number: int -- in [0, 1]
        :param logger: writer object for tensorboard
        :return: -
        """
        # figure out whether GPU or CPU is used
        # print("completing Update() with device = {}".format(device))

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        obs_full = torch.stack(obs_full).to(device)
        next_obs_full = torch.stack(next_obs_full).to(device)

        # only one agent it updated
        agent = self.maddpg_agent[agent_number]

        # ---------------------------- update local critic ---------------------------- #
        agent.critic_optimizer.zero_grad()
        # critic loss = batch-mean of [y - Q(s,a) from local network]^2
        # y = reward of this timestep + discount * Q(st+1, at+1) from target network

        # get predicted next-state actions and Q values from target models
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)

        # feed the concatenated (states + actions) directly into the *input* layer of the critic
        target_critic_input = torch.cat((next_obs_full.t(), target_actions), dim=1).to(device)
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)

        # compute the TD-target
        y = reward[agent_number].view(-1, 1).to(device) + self.discount_factor * q_next * \
            (1 - done[agent_number].view(-1, 1).to(device))

        # compute the TD-estimate
        # action = torch.cat(action, dim=1)
        action = torch.cat(action, dim=1).to(device)
        critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        q_estimate = agent.local_critic(critic_input)
        # print("q_estimate = {}".format(q_estimate))

        # compute loss on [TD-target - TD-estimate]^2
        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q_estimate, y.detach())  # y.detach() to prevent grads back

        # minimize the loss: 1)perform a backward pass and 2)update the weights
        # use autograd to compute the backward pass
        critic_loss.backward()

        # torch.nn.utils.clip_grad_norm_(agent.local_critic.parameters(), 0.5)
        # update the weights
        agent.critic_optimizer.step()

        # ---------------------------- update local actor ---------------------------- #
        # update actor local network using policy gradient
        agent.actor_optimizer.zero_grad()

        # each local actor (#1 and #2) gives its actions
        actions_predict = [self.maddpg_agent[i].local_actor(ob.to(device)).to(device) if i == agent_number else
                           self.maddpg_agent[i].local_actor(ob.to(device)).detach().to(device)  # detach() prevents grads back
                           for i, ob in enumerate(obs)]
        actions_predict = torch.cat(actions_predict, dim=1).to(device)

        # combine all the actions and observations for input to local critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        critic_input = torch.cat((obs_full.t(), actions_predict), dim=1)

        # use samples to estimate the expectation of gradient. Hence mean()
        # Deterministic Gradient Policy Theorem: gradient = expectation[Q-values]
        # pytorch by default does gradient DESCENT. Hence minus term for ASCENT
        actor_loss = -agent.local_critic(critic_input).mean()

        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.local_actor.parameters(),0.5)
        agent.actor_optimizer.step()

        # ---------------------------- Logging ---------------------------- #
        #  torch.Tensor.item() to get a Python number from a tensor containing a single value
        a_l = actor_loss.cpu().detach().item()  # prevent grads back
        c_l = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'local_critic_loss': c_l,
                            'local_actor_loss': a_l},
                           self.iter)  # number of network updates (local -> target)

    def update_targets(self):
        """soft update of critic and actor target networks for all agents"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.local_actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.local_critic, self.tau)

    def reset(self):
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.reset()
