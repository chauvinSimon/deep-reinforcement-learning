# Main code, mainly responsible for action and model updates (policy + critic )
# see ddpg.py for other details in the network

# Features:
#  - For Critic, use Huber-loss (less sensitive to outliers than the squared error loss)
#    - quadratic for small values of [target-estimate], and linear for large values


from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


class MADDPG:
    def __init__(self, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # args = in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic
        # critic input = obs_full + actions = 2*24+2+2=52
        self.maddpg_agent = [DDPGAgent(24, 16, 8, 2, 52, 32, 16),
                             DDPGAgent(24, 16, 8, 2, 52, 32, 16)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def act(self, obs_all_agents, noise=0.0):
        """get local network actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [agent.target_act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number, logger):
        """
        update critic and actor nets of given one agent
        :param samples: a list, representing the batch and containing 7 elements
        samples = [obs, obs_full, action, reward, next_obs, next_obs_full, done]
        with e.g. done = [[False, False], [False, False], False, False]] for batch_size = 3
        :param agent_number: int -- in [0, 1]
        :param logger: writer object for tensorboard
        :return: -
        """
        # see if GPU or CPU is used
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
        # critic loss = batch mean of [y - Q(s,a) from local network]^2
        # y = reward of this timestep + discount * Q(st+1, at+1) from target network

        # get predicted next-state actions and Q values from target models
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        target_critic_input = torch.cat((next_obs_full.t(), target_actions), dim=1).to(device)
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)

        # compute the TD-target
        y = reward[agent_number].view(-1, 1).to(device) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1).to(device))

        # compute the TD-estimate
        # action = torch.cat(action, dim=1)
        action = torch.cat(action, dim=1).to(device)
        critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        q = agent.critic(critic_input)

        # compute loss on [TD-target - TD-estimate]^2
        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())  # y.detach() to prevent grads back

        # minimize the loss: 1)perform a backward pass and 2)update the weights
        # use autograd to compute the backward pass
        critic_loss.backward()

        # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        # update the weights
        agent.critic_optimizer.step()

        # ---------------------------- update local actor ---------------------------- #
        # update actor local network using policy gradient
        agent.actor_optimizer.zero_grad()

        # each local actor (#1 and #2) gives its actions
        actions_predict = [self.maddpg_agent[i].actor(ob.to(device)).to(device) if i == agent_number else
                           self.maddpg_agent[i].actor(ob.to(device)).detach().to(device)  # detach() prevents grads back
                           for i, ob in enumerate(obs)]
        actions_predict = torch.cat(actions_predict, dim=1).to(device)

        # combine all the actions and observations for input to local critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        critic_input = torch.cat((obs_full.t(), actions_predict), dim=1)

        # use samples to estimate the expectation of gradient. Hence mean()
        # Deterministic Gradient Policy Theorem: gradient = expectation[Q-values]
        # pytorch by default does gradient DESCENT. Hence minus term for ASCENT
        actor_loss = -agent.critic(critic_input).mean()

        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        # ---------------------------- Logging ---------------------------- #
        #  torch.Tensor.item() to get a Python number from a tensor containing a single value
        a_l = actor_loss.cpu().detach().item()  # prevent grads back
        c_l = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': c_l,
                            'actor_loss': a_l},
                           self.iter)  # number of network updates (local -> target)

    def update_targets(self):
        """soft update of critic and actor target networks for all agents"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
