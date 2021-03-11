from sys_simulator.general import power_to_db, db_to_power
from sys_simulator.a2c.framework import ContinuousFramework, DiscreteFramework, PPOFramework
import torch
from sys_simulator.a2c import A2CLSTMDiscrete
from sys_simulator.q_learning.agents.agent import Agent as QAgent
import numpy as np


class Agent(QAgent):
    def __init__(self, max_power_db=1):
        self.bag = list()
        self.action = 0
        self.action_index = 0
        self.device =\
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_power_db = max_power_db
        self.max_power = db_to_power(max_power_db)

    def set_d2d_tx_id(self, id: str):
        self.id = id

    def act_continuous(
        self,
        framework: ContinuousFramework,
        obs: torch.TensorType,
    ):
        dist, value, sample, mu, var = framework.a2c(obs)
        clip_sample = np.clip(sample, 1e-9, 1)
        action_mag = clip_sample * self.max_power
        action_db = power_to_db(action_mag)
        self.action = action_db
        return sample, dist, value, mu, var

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


class PPOAgent:
    def __init__(self, device: torch.device):
        self.device = device

    def act(self, obs: np.ndarray, framework: PPOFramework):
        obs = torch.FloatTensor(obs)\
            .view(-1, framework.input_size).to(self.device)
        dist, value = framework.a2c(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action, log_prob, entropy, value
