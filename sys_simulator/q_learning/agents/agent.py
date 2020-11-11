from sys_simulator.devices.devices import d2d_user
import numpy as np
from sys_simulator.q_learning.q_table import QTable
from sys_simulator.parameters.parameters import AgentParameters
from typing import List


class Agent:
    """
    don't forget to set the agent actions with the set_actions method
    """

    def __init__(self, params: AgentParameters, actions: List[int]):
        self.epsilon_min = params.epsilon_min
        self.epsilon_decay = params.epsilon_decay
        self.epsilon = params.start_epsilon
        self.actions = actions
        self.action_index = 0
        self.action = 0
        self.bag = list()

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
            action_index = np.argmax(q_table.table[obs, :])
            self.action = self.actions[action_index]
            self.action_index = action_index
        else:
            action = np.random.choice(self.actions)
            self.action = action
            self.action_index = self.actions.index(action, 0)

    def get_action_tensor(self, obs, q_table):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() > self.epsilon:
            action_index = np.argmax(q_table.tensor[obs])
            self.action = self.actions[action_index]
            self.action_index = action_index
        else:
            action = np.random.choice(self.actions)
            self.action = action
            self.action_index = self.actions.index(action, 0)

    def set_action(self, action_index: int):
        self.action_index = action_index
        self.action = self.actions[action_index]

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_d2d_tx(self, d2d_tx: d2d_user):
        self.d2d_tx = d2d_tx
