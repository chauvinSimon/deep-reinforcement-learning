import numpy as np
from collections import defaultdict
import json
import pickle

# Best average reward 9.525

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.gamma = 0.9
        self.alpha = 0.1
        self.epsilon = 1  # 0.05 - constant
        self.epsilon_decay = 0.9995
        self.i_episode = 1
        self.mini_epsilon = 0.00005
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action = np.random.choice(np.arange(self.nA), 
                                  p=self.get_probs(state)) if state in self.Q else np.random.choice(np.arange(self.nA))
        return action

    def get_probs(self, state):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        best_a = np.argmax(self.Q[state])
        policy_s[best_a] = 1 - self.epsilon + (self.epsilon / self.nA)
        return policy_s

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.i_episode += 1
#         self.epsilon = max(self.epsilon_decay * self.epsilon, self.mini_epsilon)
        self.epsilon = max(1 / (self.i_episode ** 0.55), self.mini_epsilon)
        # update = get current and alternative estimates
        if not done:
#             alternative_estimate = self.sarsa(reward, next_state)
#             alternative_estimate = self.expected_sarsa(reward, next_state)
            alternative_estimate = self.sarsa_max(reward, next_state)
                    
#             state = next_state     # S <- S'
#             action = next_action   # A <- A'
        if done:
            alternative_estimate = reward
        
        current_estimate = self.Q[state][action]
        delta = alternative_estimate - current_estimate
        # update Q
        self.Q[state][action] += self.alpha * delta

    def sarsa(self, reward, next_state):
        # SARSA
        # Chooose an action At+1 following the same e-greedy policy based on current Q
        next_action = self.select_action(next_state)
        # get current and alternative estimates
        alternative_estimate = reward + self.gamma * self.Q[next_state][next_action]
        return alternative_estimate

    def sarsa_max(self, reward, next_state):
        # Q-learning (SARSA-max)
        next_q = np.max(self.Q[next_state]) if next_state in self.Q else 0
        alternative_estimate = reward + self.gamma * next_q
        return alternative_estimate

    def expected_sarsa(self, reward, next_state):
        # expected_sarsa
        next_q = np.dot(self.Q[next_state], self.get_probs(next_state)) if next_state in self.Q else 0
        alternative_estimate = reward + self.gamma * next_q
        return alternative_estimate
            

    def plot_value_function(self):
        # plot the estimated optimal state-value function
        V_sarsa = ([np.max(self.Q[key]) if key in self.Q else 0 for key in np.arange(500)])
        print(V_sarsa)
        
    def open_model(self):
#         with open('weigths.json') as data_file:
#             self.weights = json.load(data_file)
        file_name2 = 'weigths.pkl'
        pkl_file = open(file_name2)
        my_dict = pickle.load(pkl_file)
        for (key, value) in my_dict:
            self.Q[key] = value
        pkl_file.close()
        
    def save_model(self):
        # save references
        file_name = 'weigths.json'
        file_name2 = 'weigths.pkl'
        output = open(file_name2, "wb")
        pickle.dump(dict(self.Q), output)
        output.close()
#         with open(file_name, 'w') as outfile:
#             json.dump(dict(self.Q), outfile)