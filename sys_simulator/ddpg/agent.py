from torch.utils.tensorboard.writer import SummaryWriter
from sys_simulator.general import power_to_db, scale_tanh
from sys_simulator.devices.devices import d2d_node_type, d2d_user
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
        action = scale_tanh(action, self.a_min, self.a_max)
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


class SysSimAgentWriter(Agent):
    def __init__(
        self,
        a_min: float,
        a_max: float,
        exploration: str,
        device: torch.device,
    ):
        super(SysSimAgentWriter, self).__init__(a_min, a_max, exploration, device)
        self.d2d_tx = None
        self.d2d_txs = []

    def act(self, obs: ndarray, framework: Framework, writer: SummaryWriter,
            t_step: int, is_training=False, **kwargs):
        action = super(SysSimAgentWriter, self)\
            .act(obs, framework, is_training, **kwargs)
        writer.add_scalar('Average Actor output',
                          np.mean(action.detach().numpy()), t_step)
        action = power_to_db(action)
        return action


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

    def act(self, obs: ndarray, framework: Framework, is_training=False, **kwargs):
        action = super(SysSimAgent, self)\
            .act(obs, framework, is_training, **kwargs)
        action = power_to_db(action)
        return action



class SurrogateAgent:
    """Agent that will just interface the environment with the `SysSimAgent`.
    """

    def __init__(self):
        self.action = 1e-9
        self.d2d_tx = d2d_user(0, d2d_node_type.TX)

    def set_d2d_tx(self, d2d_tx: d2d_user):
        self.d2d_tx = d2d_tx

    def set_d2d_tx_id(self, id: str):
        self.id = id

    def set_action(self, action: float):
        """Receives `action`. Action must be in dB.
        """
        self.action = action
