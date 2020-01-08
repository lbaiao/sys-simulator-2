import sys
import os
lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from parameters.parameters import LearningParameters
import numpy as np

class QTable:
    def __init__(self, num_actions: int, params: LearningParameters):
        self.table = np.zeros((2, num_actions))
        self.gamma = params.gamma
        self.alpha = params.alpha

    # calculates Q-table values
    def learn(self, obs, action_index, reward, next_obs):
        deltaQ = reward + self.gamma*np.max(self.table[next_obs]) - self.table[obs, action_index]
        self.table[obs, action_index] = self.table[obs,action_index] + self.alpha*deltaQ


class DistributedQTable:
    def __init__(self, num_actions: int, params: LearningParameters):
        self.table = np.zeros((2, num_actions))
        self.gamma = params.gamma
        self.alpha = params.alpha

    # calculates Q-table values
    def learn(self, obs, action_index, reward, next_obs):
        deltaQ = reward + self.gamma*np.max(self.table[next_obs]) - self.table[obs, action_index]
        q_value = self.table[obs,action_index] + self.alpha*deltaQ
        if q_value > self.table[obs, action_index]:
            self.table[obs, action_index] = q_value
        