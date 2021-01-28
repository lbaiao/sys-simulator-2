from sys_simulator.a2c.agent import Agent
from sys_simulator.general import db_to_power
import numpy as np
import torch

from typing import List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def centralized_reward(sinr_mue: float, sinr_d2ds: List[float],
                       *args, **kwargs):
    mue_contrib = np.log2(1 + sinr_mue)
    d2d_contrib = sum([np.log2(1 + s) for s in sinr_d2ds])
    reward = mue_contrib + d2d_contrib
    return reward, mue_contrib, d2d_contrib


def mod_reward(sinr_mue: float, sinr_d2ds: List[float],
               state: int, *args, **kwargs):
    mue_contrib = np.log2(1 + sinr_mue)
    d2d_contrib = sum([np.log2(1 + s) for s in sinr_d2ds])
    if state:
        reward = mue_contrib + d2d_contrib
    else:
        reward = -1
    return reward, mue_contrib, d2d_contrib


def dis_reward(sinr_mue: float, sinr_d2ds: List[float], state: int,
               C: float, *args, **kwargs):
    mue_contrib = np.log2(1 + sinr_mue)
    d2d_contrib = sum([np.log2(1 + s) for s in sinr_d2ds])
    rewards = -1 * np.ones(len(sinr_d2ds))
    if state:
        for i in range(len(sinr_d2ds)):
            rewards[i] = 1/C * np.log2(1 + sinr_d2ds[i])
    return rewards, mue_contrib, d2d_contrib


def dis_reward_tensor(sinr_mue: float, sinr_d2ds: List[float],
                      state: int, C: float, *args, **kwargs):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    mue_contrib = torch.log2(1 + torch.tensor(sinr_mue, device=device))
    sinr_d2ds = torch.tensor(sinr_d2ds, device=device)
    d2d_contrib = torch.sum(torch.log2(1 + sinr_d2ds))
    rewards = -1 * torch.ones(len(sinr_d2ds))
    if state:
        for i in range(len(sinr_d2ds)):
            rewards[i] = 1/C * torch.log2(1 + sinr_d2ds[i])
    return rewards, mue_contrib, d2d_contrib


def dis_reward_tensor_mod(sinr_mue: float, sinr_d2ds: List[float],
                          state: int, C: float, *args, **kwargs):
    penalty = kwargs['penalty']
    device = torch.device('cuda')
    mue_contrib = torch.log2(1 + torch.tensor(sinr_mue, device=device))
    sinr_d2ds = torch.tensor(sinr_d2ds, device=device)
    d2d_contrib = torch.sum(torch.log2(1 + sinr_d2ds))
    # d2d_contrib = torch.sum(torch.tensor([torch.log2(1 + s)
    # for s in sinr_d2ds], device=device))
    rewards = -penalty * torch.ones(len(sinr_d2ds))
    if state:
        for i in range(len(sinr_d2ds)):
            rewards[i] = 1/C * torch.log2(1 + sinr_d2ds[i])
    return rewards, mue_contrib, d2d_contrib


def dis_reward_tensor2(sinr_mue: float, sinr_d2ds: List[float],
                       state: int, C: float, *args, **kwargs):
    device = torch.device('cuda')
    mue_contrib = torch.log2(1 + torch.tensor(sinr_mue, device=device))
    sinr_d2ds = torch.tensor(sinr_d2ds, device=device)
    d2d_contrib = torch.sum(torch.log2(1 + sinr_d2ds))
    rewards = -10/C * torch.ones(len(sinr_d2ds))
    if state:
        for i in range(len(sinr_d2ds)):
            rewards[i] = 1/C * torch.log2(1 + sinr_d2ds[i])
    return rewards, mue_contrib, d2d_contrib


def dis_reward_tensor_db(sinr_mue_db: float, sinr_d2ds_db: List[float],
                         state: bool, C: float, *args, **kwargs):
    sinr_mue = db_to_power(sinr_mue_db)
    sinr_d2ds = db_to_power(np.array(sinr_d2ds_db))
    mue_contrib = torch.log2(1 + torch.tensor(sinr_mue, device=device))
    sinr_d2ds = torch.tensor(sinr_d2ds, device=device)
    d2d_contrib = torch.sum(torch.log2(1 + sinr_d2ds))
    rewards = -1 * torch.ones(len(sinr_d2ds))
    if state:
        # # debugging
        # print('debug')
        for i in range(len(sinr_d2ds)):
            rewards[i] = 1/C * torch.log2(1 + sinr_d2ds[i])
    return rewards, mue_contrib, d2d_contrib


def index_reward_db(agents: List[Agent], state: bool):
    if state:
        rewards = np.array([a.action_index for a in agents])
    else:
        rewards = -10 * np.ones(len(agents))
    return rewards


def dis_reward_tensor_portela(sinr_mue: float, sinr_d2ds: List[float],
                              state: int, C: float,  distances_1: List[float],
                              distances_2: List[float], mu: float,
                              *args, **kwargs):
    device = torch.device('cuda')

    d1_max = np.max(distances_1)
    d2_max = np.max(distances_2)

    coefs_1 = distances_1 / d1_max
    coefs_2 = distances_2 / d2_max
    betas = mu * (coefs_1 + coefs_2)
    betas = torch.tensor(betas, device=device).double()

    mue_contrib = torch.log2(1 + torch.tensor(sinr_mue, device=device))
    sinr_d2ds = torch.tensor(sinr_d2ds, device=device).double()
    d2d_contrib = torch.sum(torch.log2(1 + sinr_d2ds))
    # d2d_contrib = torch.sum(torch.tensor([torch.log2(1 + s)
    # for s in sinr_d2ds], device=device))
    rewards = torch.ones(len(sinr_d2ds))
    if state:
        # for i in range(len(sinr_d2ds)):
        #     rewards[i] = betas[i] *  1/C * torch.log2(1 + sinr_d2ds[i])
        rewards = betas * 1/C * torch.log2(1 + sinr_d2ds)
    return rewards, mue_contrib, d2d_contrib


def dis_reward_tensor_portela_inverse(
    sinr_mue: float, sinr_d2ds: List[float], state: int, C: float,
    distances_1: List[float], distances_2: List[float],
    mu: float, *args, **kwargs
):
    device = torch.device('cuda')

    d1_max = np.max(distances_1)
    d2_max = np.max(distances_2)

    coefs_1 = distances_1 / d1_max
    coefs_2 = distances_2 / d2_max
    betas = mu * 1 / (coefs_2 + coefs_1)
    betas = torch.tensor(betas, device=device).double()

    mue_contrib = torch.log2(1 + torch.tensor(sinr_mue, device=device))
    sinr_d2ds = torch.tensor(sinr_d2ds, device=device).double()
    d2d_contrib = torch.sum(torch.log2(1 + sinr_d2ds))
    # d2d_contrib = torch.sum(torch.tensor([torch.log2(1 + s)
    # for s in sinr_d2ds], device=device))
    rewards = torch.ones(len(sinr_d2ds))
    if state:
        # for i in range(len(sinr_d2ds)):
        #     rewards[i] = betas[i] *  1/C * torch.log2(1 + sinr_d2ds[i])
        rewards = betas * 1/C * torch.log2(1 + sinr_d2ds)
    return rewards, mue_contrib, d2d_contrib
