from agent import Agent
from monitor import interact
import gym
import gym.spaces
import numpy as np

env = gym.make('Taxi-v2')
agent = Agent()
# agent.open_model()

# Discrete(6)
# Discrete(500)
avg_rewards, best_avg_reward = interact(env, agent)
# Best average reward 7.533
agent.save_model()
# agent.plot_value_function()