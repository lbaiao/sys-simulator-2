from sys_simulator.general.general import power_to_db
from sys_simulator.a2c.framework import ContinuousFramework, DiscreteFramework
import torch
from sys_simulator.a2c import A2CLSTMDiscrete, ActorCritic
from sys_simulator.q_learning.agents.agent import Agent as QAgent


class Agent(QAgent):
    def __init__(self):
        self.bag = list()
        self.action = 0
        self.action_index = 0
        self.device =\
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_d2d_tx_id(self, id: str):
        self.id = id

    def act_continuous(
        self,
        framework: ContinuousFramework,
        obs: torch.TensorType
    ):
        dist, value, action = framework.a2c(obs)
        self.action = action
        return self.action, dist, value

    def act_discrete(
        self,
        framework: DiscreteFramework,
        obs: torch.TensorType
    ):
        dist, value, _ = framework.a2c(obs)
        self.action_index = dist.sample()
        # for debugging
        # if self.action_index > 4:
        #     print('problems')
        return self.action_index, dist, value

    def act_lstm_discrete(self, a2c: A2CLSTMDiscrete,
                          obs, actor_hidden_h, actor_hidden_c):
        dist, value, actor_hidden_h, actor_hidden_c = \
            a2c(obs, actor_hidden_h, actor_hidden_c)
        self.action_index = dist.sample()
        return self.action_index, dist, value, actor_hidden_h, actor_hidden_c

    def get_action(self):
        return self.action

    def set_action(self, action):
        self.action = action
