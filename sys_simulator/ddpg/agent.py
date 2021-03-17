from sys_simulator.general.ou_noise import OUNoise
from types import MethodType
import numpy as np
from numpy.random import normal
import torch
from sys_simulator.ddpg.framework import Framework
from numpy import ndarray


class Agent:
    a_min: float
    a_max: float
    exploration: str
    explore: MethodType
    device: torch.device

    def __init__(
        self,
        a_min: float,
        a_max: float,
        exploration: str,
        device: torch.device,
    ):
        self.a_min = a_min
        self.a_max = a_max
        self.device = device
        if exploration == 'gauss':
            self.explore = self.gauss_explore
        elif exploration == 'ou':
            self.explore = self.ou_explore
        else:
            raise Exception('Invalid exploration.')

    def act(self, obs: ndarray, framework: Framework,
            is_training=False, **kwargs):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = framework.actor(obs)
        action = action.detach().cpu()
        action = action.squeeze(0)
        if is_training:
            action = self.explore(action=action, **kwargs)
        action = action * (self.a_max - self.a_min) / 2
        action = np.clip(action, self.a_min, self.a_max)
        return action

    def gauss_explore(self, **kwargs):
        action = kwargs['action']
        action += normal(loc=0, scale=1)
        return action

    def ou_explore(self, **kwargs):
        ou: OUNoise = kwargs['ou']
        action = kwargs['action']
        step = kwargs['step']
        mu = ou.get_action(action, step)
        return mu


class SysSimAgent(Agent):
    def __init__(
        self,
        a_min: float,
        a_max: float,
        exploration: str,
        device: torch.device,
    ):
        super(SysSimAgent, self).__init__(a_min, a_max, exploration, device)
        self.d2d_tx = None
        self.d2d_txs = []
