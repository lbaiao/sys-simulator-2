from math import pi
import numpy as np
from copy import deepcopy
from time import time
from sys_simulator.plots import plot_positions, plot_trajectories
from sys_simulator.ddpg.agent import SurrogateAgent, SysSimAgent
from sys_simulator.parameters.parameters import EnvironmentParameters
from sys_simulator.q_learning.environments.completeEnvironment12 import CompleteEnvironment12
from sys_simulator.channels import BANChannel, UrbanMacroNLOSWinnerChannel
import sys_simulator.general as gen
from sys_simulator.general import print_evaluating, random_seed
from sys_simulator.ddpg.framework import Framework
import torch
from torch.utils.tensorboard.writer import SummaryWriter


# parameters
ALGO_NAME = 'ddpg'
FRAMEWORK_PATH = 'D:\\Dev/sys-simulator-2/data/ddpg/script7/20210429-181155/last_model.pt'   # noqa
n_mues = 1  # number of mues
n_rb = n_mues   # number of RBs
carrier_frequency = 2.4  # carrier frequency in GHz
bs_radius = 1000  # bs radius in m
rb_bandwidth = 180*1e3  # rb bandwidth in Hz
d2d_pair_distance = 5  # d2d pair distance in m
device_height = 1.5  # mobile devices height in m
bs_height = 25  # BS antenna height in m
p_max = 40  # max tx power in dBm
noise_power = -116  # noise power per RB in dBm
bs_gain = 17    # macro bs antenna gain in dBi
user_gain = 4   # user antenna gain in dBi
sinr_threshold_train = 6  # mue sinr threshold in dB for training
mue_margin = 6
MIN_D2D_PAIR_DISTANCE = 1.5
MAX_D2D_PAIR_DISTANCE = 15
# conversions from dBm to dB
p_max = p_max - 30
noise_power = noise_power - 30
# env parameters
RND_SEED = True
SEED = 42
CHANNEL_RND = True
C = 8  # C constant for the improved reward function
ENVIRONMENT_MEMORY = 2
MAX_NUMBER_OF_AGENTS = 2
REWARD_PENALTY = 1.5
DELTA_T = .5
# q-learning parameters
# training
REWARD_FUNCTION = 'jain'
STATES_OPTIONS = ['sinrs', 'positions', 'channels']
MOTION_MODEL = 'forward'
EVAL_STEPS = 700

# writer
filename = gen.path_leaf(__file__)
filename = filename.split('.')[0]
dir_path = f'data/{ALGO_NAME}/{filename}'
data_path, _ = gen.make_dir_timestamp(dir_path)
writer = SummaryWriter(f'{data_path}/tensorboard')
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
    small_sigma=4.0, sigma=8.0
)
ref_env = CompleteEnvironment12(
    env_params,
    channel_to_bs,
    channel_to_devices,
    reward_penalty=REWARD_PENALTY,
    memory=ENVIRONMENT_MEMORY,
    bs_height=bs_height,
    reward_function=REWARD_FUNCTION,
    states_options=STATES_OPTIONS,
    memories_capacity=int(1e3),
    dt=DELTA_T
)
a_min = -90
a_max = 60
a_offset = -10
# a_min = 0 + 1e-9
# a_max = db_to_power(p_max - 10)
action_size = MAX_NUMBER_OF_AGENTS
framework: Framework = torch.load(FRAMEWORK_PATH)
framework.actor.eval()
central_agent_test = SysSimAgent(a_min, a_max, 'perturberd',
                                 torch_device, a_offset=a_offset)
surr_agents = [SurrogateAgent() for _ in range(MAX_NUMBER_OF_AGENTS)]
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
    env = deepcopy(ref_env)
    env.set_scenario(pairs_positions, mue_position, surr_agents, motion_model=MOTION_MODEL)  # noqa
    # set directions
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
    devices = env.get_devices()
    trajectories = {d.id: [d.position] for d in devices}
    writer.add_figure('Devices positions', positions_fig)
    d2d_sinrs = []
    while step < EVAL_STEPS:
        obs, _, _, _ = env.step(surr_agents)
        now = (time() - start) / 60
        print_evaluating(step, now, EVAL_STEPS)
        done = False
        i = 0
        actions = central_agent_test.act(obs, framework, False)
        # db_actions = power_to_db(actions)
        db_actions = actions
        for j, agent in enumerate(surr_agents):
            agent.set_action(db_actions[j].item())
        next_obs, reward, done, _ = env.step(surr_agents)
        framework.replay_memory.push(obs, actions, reward, next_obs,
                                     done)
        obs = next_obs
        i += 1
        step += 1
        writer.add_scalar('3. Eval - Rewards', reward, step)
        sinrs = [a.d2d_tx.sinr for a in surr_agents]
        d2d_sinrs.append(sinrs)
        sinrs = {f'device {i}': s for i, s in enumerate(sinrs)}
        writer.add_scalars('3. Eval - SINRs [dB]', sinrs, step)
        writer.add_scalars(
            '3. Eval - Transmission powers [dBW]',
            {f'device {i}': a for i, a in enumerate(db_actions)},
            step
        )
        writer.add_scalar('3. Eval - MUE Tx Power [dB]',
                          env.mue.tx_power, step)
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
