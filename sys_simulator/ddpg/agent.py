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
        mu = framework.actor(obs)
        mu = mu.detach().cpu().numpy()[0, 0]
        if is_training:
            self.explore(action=mu, **kwargs)
        # action = mu * (self.a_max - self.a_min) / 2
        # action = np.clip(mu, self.a_min, self.a_max)
        action = mu
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
