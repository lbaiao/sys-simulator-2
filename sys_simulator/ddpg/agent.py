import numpy as np
from numpy.random import normal
import torch
from sys_simulator.ddpg.framework import Framework
from numpy import ndarray


class Agent:
    def __init__(self, a_min: float, a_max: float, device: torch.device):
        self.a_min = a_min
        self.a_max = a_max
        self.device = device

    def act(self, obs: ndarray, framework: Framework, is_training=False):        
        obs = torch.Tensor(obs).to(self.device)
        mu = framework.ddpg.actor(obs).item()
        if is_training:
            mu += normal(loc=0, scale=1)
        action = np.clip(mu, self.a_min, self.a_max)
        return action
