import sys
import os
# lucas_path = os.environ['LUCAS_PATH']
# sys.path.insert(1, lucas_path)

from sys_simulator.parameters.parameters import LearningParameters
import numpy as np
import torch


class QTable:
    def __init__(self, num_states: int, num_actions: int, params: LearningParameters):
        self.table = np.zeros((num_states, num_actions))
        self.gamma = params.gamma
        self.alpha = params.alpha

    # calculates Q-table values
    def learn(self, obs, action_index, reward, next_obs):
        deltaQ = reward + self.gamma*np.max(self.table[next_obs]) - self.table[obs, action_index]
        self.table[obs, action_index] = self.table[obs,action_index] + self.alpha*deltaQ


class QTensor:
    def __init__(self, tensor: np.array, params: LearningParameters):
        self.tensor = tensor
        self.gamma = params.gamma
        self.alpha = params.alpha


    # calculates Q-table values
    def learn(self, obs, action, reward, next_obs):
        deltaQ = reward + self.gamma*self.tensor[next_obs].max() - self.tensor[(*obs, action)]
        self.tensor[obs] += self.alpha*deltaQ


class QTableTorch:
    def __init__(self, num_states: int, num_actions: int, params: LearningParameters):
        self.device = torch.device('cuda')
        self.table = torch.zeros((num_states, num_actions), device=self.device)
        self.gamma = params.gamma
        self.alpha = params.alpha

    # calculates Q-table values
    def learn(self, obs, action_index, reward, next_obs):
        deltaQ = reward + self.gamma*torch.max(self.table[next_obs]) - self.table[obs, action_index]
        self.table[obs, action_index] = self.table[obs,action_index] + self.alpha*deltaQ

class DistributedQTable(QTable):
    def __init__(self, num_states: int, num_actions: int, params: LearningParameters):
        super(DistributedQTable, self).__init__(num_states, num_actions, params)

    # calculates Q-table values
    def learn(self, obs, action_index, reward, next_obs):
        deltaQ = reward + self.gamma*np.max(self.table[next_obs]) - self.table[obs, action_index]
        q_value = self.table[obs,action_index] + self.alpha*deltaQ
        if q_value > self.table[obs, action_index]:
            self.table[obs, action_index] = q_value

class ModDistributedQTable(QTable):
    def __init__(self, num_states: int, num_actions: int, params: LearningParameters):
        super(ModDistributedQTable, self).__init__(num_states, num_actions, params)

    # calculates Q-table values
    def learn(self, obs, action_index, reward, next_obs):
        deltaQ = reward + self.gamma*np.max(self.table[next_obs]) - self.table[obs, action_index]
        q_value = self.table[obs,action_index] + self.alpha*deltaQ
        self.table[obs, action_index] = q_value


class DistributedQTensor:
    """
    table is now a tensor. obs is a now a two-element-tuple
    """
    def __init__(self, num_actions: int, num_mue_states: int, params: LearningParameters):
        self.tensor = np.zeros((num_actions, num_mue_states, num_actions))
        self.gamma = params.gamma
        self.alpha = params.alpha

    # calculates Q-table values
    def learn(self, obs, action_index, reward, next_obs):
        deltaQ = reward + self.gamma*np.max(self.tensor[next_obs]) - self.tensor[(obs[0], obs[1], action_index)]
        q_value = self.tensor[(obs[0], obs[1], action_index)] + self.alpha*deltaQ
        if q_value > self.tensor[(obs[0], obs[1], action_index)]:
            self.tensor[(obs[0], obs[1], action_index)] = q_value
        