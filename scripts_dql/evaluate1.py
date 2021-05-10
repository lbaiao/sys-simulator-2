from copy import deepcopy
from math import pi
from time import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sys_simulator.channels import BANChannel, UrbanMacroNLOSWinnerChannel
from sys_simulator.ddpg.parameters_noise import AdaptiveParamNoiseSpec
from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from sys_simulator.dqn.externalDQNFramework import ExternalDQNFramework
import sys_simulator.general as gen
from sys_simulator.general import (
    load_with_pickle,
    power_to_db,
    print_evaluating,
    print_stuff_ddpg,
    random_seed,
)
from sys_simulator.general.actions_discretizations import db_five, db_six, db_ten
from sys_simulator.noises.decaying_gauss_noise import DecayingGaussNoise
from sys_simulator.noises.ou_noise import OUNoise2, SysSimOUNoise
from sys_simulator.parameters.parameters import (
    DQNAgentParameters,
    EnvironmentParameters,
)
from sys_simulator.plots import plot_env_states, plot_positions, plot_trajectories
from sys_simulator.q_learning.environments.completeEnvironment12 import (
    CompleteEnvironment12,
)

# Uses CompleteEnvironment12
# Centralized Learning-Distributed Execution
# NN Parameters noise.
# Similar to script5.
# Trains a central agent for a fixed amount of devices.
# There are different channels to the BS and to the devices.
# There are multiple episodes, where the devices are distributed
# in different positions, and there are different motion models.
# Single episodes convergence. The states are in linear scale.
n_mues = 1  # number of mues
n_d2d = 2  # number of d2d pairs
n_rb = n_mues   # number of RBs
carrier_frequency = 2.4  # carrier frequency in GHz
bs_radius = 1000  # bs radius in m
rb_bandwidth = 180*1e3  # rb bandwidth in Hz
device_height = 1.5  # mobile devices height in m
bs_height = 25  # BS antenna height in m
p_max = 40  # max tx power in dBm
noise_power = -116  # noise power per RB in dBm
bs_gain = 17    # macro bs antenna gain in dBi
user_gain = 4   # user antenna gain in dBi
sinr_threshold_train = 6  # mue sinr threshold in dB for training
mue_margin = 6  # mue margin in dB
MIN_D2D_PAIR_DISTANCE = 1.5
MAX_D2D_PAIR_DISTANCE = 15
# conversions from dBm to dB
p_max = p_max - 30
noise_power = noise_power - 30
p_min = -90
# env parameters
RND_SEED = True
SEED = 42
CHANNEL_RND = True
C = 8  # C constant for the improved reward function
ENVIRONMENT_MEMORY = 2
MAX_NUMBER_OF_AGENTS = 2
REWARD_PENALTY = 1.5
N_STATES_BINS = 100
DELTA_T = .5
# q-learning parameters
# training
ALGO_NAME = 'dql'
DATA_PATH = 'D:\\Dev/sys-simulator-2/data\\dql\\script52\\20210427-141357'  # noqa
FRAMEWORK_PATH = f'{DATA_PATH}/last_model.pt'
ENV_PATH = f'{DATA_PATH}/env.pickle'
REWARD_FUNCTION = 'multi_agent_continuous'
STATES_OPTIONS = ['sinrs', 'positions', 'channels', 'powers']
MOTION_MODEL = 'forward'
STATES_FUNCTION = 'multi_agent'
EVAL_STEPS = 700
REPLAY_MEMORY_SIZE = int(10E3)
BATCH_SIZE = 128
GAMMA = .5
EPSILON_INITIAL = 1
EPSILON_MIN = .01
EPSILON_DECAY = 1 / (.1 * 10000)

# writer
filename = gen.path_leaf(__file__)
filename = filename.split('.')[0]
dir_path = f'data/{ALGO_NAME}/{filename}'
data_path, _ = gen.make_dir_timestamp(dir_path)
writer = SummaryWriter(f'{data_path}/tensorboard')
framework: ExternalDQNFramework = torch.load(FRAMEWORK_PATH)
framework.policy_net.eval()
if RND_SEED:
    random_seed(SEED)
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env_params = EnvironmentParameters(
    rb_bandwidth, None, p_max, noise_power,
    bs_gain, user_gain, sinr_threshold_train,
    n_mues, MAX_NUMBER_OF_AGENTS, n_rb, bs_radius,
    c_param=C, mue_margin=mue_margin,
    min_d2d_pair_distance=MIN_D2D_PAIR_DISTANCE,
    max_d2d_pair_distance=MAX_D2D_PAIR_DISTANCE
)
channel_to_devices = BANChannel(rnd=CHANNEL_RND)
channel_to_bs = UrbanMacroNLOSWinnerChannel(
    rnd=CHANNEL_RND, f_c=carrier_frequency, h_bs=bs_height, h_ms=device_height,
    small_sigma=8.0, sigma=8.0
)
env: CompleteEnvironment12 = load_with_pickle(ENV_PATH)
env.dt = DELTA_T
action_size = MAX_NUMBER_OF_AGENTS
env_state_size = env.state_size()
# actions = db_five(p_min, p_max)
actions = db_six(p_min, p_max)
# actions = db_ten(p_min, p_max)
NUMBER_OF_ACTIONS = len(actions)
agent_params = DQNAgentParameters(
    EPSILON_MIN, EPSILON_DECAY, EPSILON_INITIAL, REPLAY_MEMORY_SIZE,
    BATCH_SIZE, GAMMA
)
agents = [ExternalDQNAgent(agent_params, actions)
          for _ in range(MAX_NUMBER_OF_AGENTS)]
pairs_positions = [
    ((-1, 0, device_height), (-1, 5, device_height)),
    ((-1, 0, device_height), (-1, -5, device_height)),
]
pairs_directions = [
    (2*pi/3, 2*pi/3),
    (4*pi/3, 4*pi/3),
]
mue_position = (900, 0, device_height)
mue_direction = pi
n_agents = len(pairs_positions)


def evaluate(start: float, writer: SummaryWriter):
    step = 0
    mue_availability = []
    env.reset()
    env.set_scenario(pairs_positions, mue_position, agents, motion_model=MOTION_MODEL)  # noqa
    for (tx, rx), (d_tx, d_rx) in zip(env.d2d_pairs, pairs_directions):
        tx.motion_model.direction = d_tx
        rx.motion_model.direction = d_rx
    env.mue.motion_model.direction = mue_direction
    positions_fig = plot_positions(
        env.bs, [env.mue],
        [p[0] for p in env.d2d_pairs],
        [p[1] for p in env.d2d_pairs],
        False
    )
    writer.add_figure('3. Eval - Initial position',
                      positions_fig, step)
    devices = env.get_devices()
    trajectories = {d.id: [d.position] for d in devices}
    d2d_sinrs = []
    while step < EVAL_STEPS:
        obs, _, _, _ = env.step(agents)
        now = (time() - start) / 60
        print_evaluating(step, now, EVAL_STEPS)
        done = False
        i = 0
        past_actions = np.zeros(len(agents))
        with torch.no_grad():
            for j, agent in enumerate(agents):
                agent.act(framework, obs[j])
                past_actions[j] = agent.action_index
        next_obs, reward, done, _ = env.step(agents)
        obs = next_obs
        i += 1
        step += 1
        rewards = {f'device {i}': r for i, r in enumerate(reward)}
        writer.add_scalars('3. Eval - Rewards', rewards, step)
        sinrs = [a.d2d_tx.sinr for a in agents]
        d2d_sinrs.append(sinrs)
        sinrs = {f'device {i}': s for i, s in enumerate(sinrs)}
        writer.add_scalars('3. Eval - SINRs [dB]', sinrs, step)
        writer.add_scalar('3. Eval - MUE Tx Power [dB]',
                          env.mue.tx_power, step)
        writer.add_scalars(
            '3. Eval - Transmission powers [dBW]',
            {f'device {i}': a.action for i, a in enumerate(agents)},
            step
        )
        writer.add_scalar(
            '3. Eval - MUE SINR [dB]', env.mue.sinr, step)
        mue_success = int(env.mue.sinr > env.params.sinr_threshold)
        mue_availability.append(mue_success)
        writer.add_scalar('3. Eval - MUE success', mue_success, step)
        for d in env.get_devices():
            trajectories[d.id].append(d.position)
    mue_availability = np.mean(mue_availability)
    d2d_sinrs = np.mean(d2d_sinrs, axis=0)
    writer.add_text('3. Eval - Average MUE availability',
                    str(mue_availability), step)
    for i, s in enumerate(d2d_sinrs):
        writer.add_text(f'3. Eval - Average D2D {i} SINR', str(s), step)
    writer.add_figure('3. Eval - Trajectories',
                      plot_trajectories(env, trajectories), step)


if __name__ == '__main__':
    start = time()
    evaluate(start, writer)
