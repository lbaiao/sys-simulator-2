import os
import sys
lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

import numpy as np
from q_learning.q_table import QTable
from parameters.parameters import AgentParameters

class Agent:
    """
    don't forget to set the agent actions with the set_actions method
    """
    def __init__(self, params: AgentParameters, actions):
        self.epsilon_min = params.epsilon_min
        self.epsilon_decay = params.epsilon_decay
        self.epsilon = params.start_epsilon
        self.actions = actions

    def set_q_table(self, q_table: QTable):
        self.q_table = q_table

    def set_d2d_tx_id(self, id: str):
        self.id = id

    def set_actions(self, actions):
        self.actions = actions

    def get_action(self, obs, q_table):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() > self.epsilon:
            action_index = np.argmax(q_table.table[obs,:])
            self.action = self.actions[action_index]
            self.action_index = action_index
        else:
            action = np.random.choice(self.actions)
            self.action = action
            self.action_index = self.actions.index(action, 0)

    def set_action(self, action_index: int):
        self.action_index = action_index
        self.action = self.actions[action_index]
        