from copy import copy
from types import MethodType

import numpy as np
from numpy import ndarray
from numpy.random import normal
import torch
from torch.autograd import Variable
from torch.utils.tensorboard.writer import SummaryWriter

from sys_simulator.ddpg.framework import Framework, PerturberdFramework
from sys_simulator.devices.devices import d2d_node_type, d2d_user
from sys_simulator.general import power_to_db, scale_tanh
from sys_simulator.noises.ou_noise import OUNoise


class Agent:
    a_min: float
    a_max: float
    exploration: str
    explore: MethodType
    device: torch.device
    nn_output: torch.FloatTensor

    def __init__(
        self,
        a_min: float,
        a_max: float,
        exploration: str,
        device: torch.device,
        a_offset=0.0
    ):
        self.a_min = a_min
        self.a_max = a_max
        self.device = device
        self.a_offset = a_offset
        if exploration == 'gauss':
            self.explore = self.gauss_explore
        elif exploration == 'ou':
            self.explore = self.ou_explore
        elif exploration == 'perturberd':
            self.explore = self.parameters_noise_explore
        else:
            raise Exception('Invalid exploration.')

    def act(self, obs: ndarray, framework: Framework,
            is_training=False, **kwargs):
        if is_training:
            action = self.explore(obs, framework, **kwargs)
        else:
            action = self.call_framework(obs, framework)
        # action = scale_tanh(action, self.a_min, self.a_max)
        # action = action/200
        action += self.a_offset
        # action = np.clip(action, self.a_min, self.a_max)
        return action

    def call_framework(self, obs: ndarray, framework: Framework):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = framework.actor(obs)
        action = action.detach().cpu()
        action = action.squeeze(0)
        self.nn_output = copy(action)
        return action

    def gauss_explore(self, obs: ndarray, framework: Framework, **kwargs):
        action = self.call_framework(obs, framework)
        noise = normal(loc=0, scale=0.05)
        action += noise
        return action

    def ou_explore(self, obs: ndarray, framework: Framework, **kwargs):
        action = self.call_framework(obs, framework)
        ou: OUNoise = kwargs['ou']
        step = kwargs['step']
        mu = ou.get_action(action, step)
        return mu

    def parameters_noise_explore(
        self, obs: ndarray,
        framework: Framework, **kwargs
    ):
        noise = kwargs.pop('noise', None)
        param_noise = kwargs.pop('param_noise', None)
        obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        obs = Variable(obs, requires_grad=False)\
            .type(torch.FloatTensor).to(self.device)
        framework.actor.eval()
        framework.actor_perturberd.eval()
        if param_noise is not None:
            action = framework.actor_perturberd(obs)
        else:
            action = framework.actor(obs)
        action = action.data[0].cpu().numpy()
        self.nn_output = copy(action)
        framework.actor.train()
        if noise is not None:
            action = action + noise
        return action


class SysSimAgentWriter(Agent):
    def __init__(
        self,
        a_min: float,
        a_max: float,
        exploration: str,
        device: torch.device,
        a_offset=0.0
    ):
        super(SysSimAgentWriter, self).__init__(
            a_min, a_max, exploration, device, a_offset)
        self.d2d_tx = None
        self.d2d_txs = []

    def act(self, obs: ndarray, framework: Framework, writer: SummaryWriter,
            t_step: int, is_training=False, **kwargs):
        action = super(SysSimAgentWriter, self)\
            .act(obs, framework, is_training, **kwargs)
        writer.add_scalars(
            '1. Training - Actor NN outputs',
            {f'device {i}': a for i, a in enumerate(self.nn_output)},
            t_step
        )
        writer.add_scalars(
            '1. Training - Actor agent outputs',
            {f'device {i}': a for i, a in enumerate(action)},
            t_step
        )
        return action


class SysSimAgent(Agent):
    def __init__(
        self,
        a_min: float,
        a_max: float,
        exploration: str,
        device: torch.device,
        a_offset=0.0
    ):
        super(SysSimAgent, self).__init__(
            a_min, a_max, exploration,
            device, a_offset
        )
        self.d2d_tx = None
        self.d2d_txs = []

    def act(self, obs: ndarray, framework: Framework, is_training=False, **kwargs):
        action = super(SysSimAgent, self)\
            .act(obs, framework, is_training, **kwargs)
        # action = power_to_db(action)
        return action


class SurrogateAgent:
    """Agent that will just interface the environment with the `SysSimAgent`.
    """

    def __init__(self):
        self.action = 1e-9
        self.d2d_tx = d2d_user(0, d2d_node_type.TX)
        self.nn_output = 1e-9

    def set_d2d_tx(self, d2d_tx: d2d_user):
        self.d2d_tx = d2d_tx

    def set_d2d_tx_id(self, id: str):
        self.id = id

    def set_action(self, action: float):
        """Receives `action`. Action must be in dB.
        """
        self.action = action

    def set_nn_output(self, nn_output: float):
        self.nn_output = nn_output
