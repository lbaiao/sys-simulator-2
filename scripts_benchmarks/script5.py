# Uses CompleteEnvironment12
# Choose the power levels according to an uniform distribution.
# Benchmark for discrete power levels.
# Trains a central agent for a fixed amount of devices.
# There are different channels to the BS and to the devices.
# There are multiple episodes, where the devices are distributed
# in different positions, and there are different motion models.
from copy import deepcopy
from random import random
from shutil import copyfile
from sys_simulator.general.actions_discretizations import db_five, db_six, db_ten
from sys_simulator.dqn.externalDQNFramework import ExternalDQNFramework
from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from time import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sys_simulator.channels import BANChannel, UrbanMacroNLOSWinnerChannel
from sys_simulator.ddpg.parameters_noise import (
    AdaptiveParamNoiseSpec,
)
import sys_simulator.general as gen
from sys_simulator.general import (
    power_to_db,
    print_evaluating,
    print_stuff_ddpg,
    random_seed,
    save_with_pickle,
)
from sys_simulator.parameters.parameters import DQNAgentParameters, EnvironmentParameters
from sys_simulator.plots import plot_env_states, plot_positions, plot_trajectories
from sys_simulator.q_learning.environments.completeEnvironment12 import (
    CompleteEnvironment12,
)

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
DELTA_T = 1e-3
# q-learning parameters
# training
ALGO_NAME = 'benchmark'
REWARD_FUNCTION = 'multi_agent_continuous'
STATES_OPTIONS = ['sinrs', 'positions', 'channels', 'powers']
MOTION_MODEL = 'random'
# MOTION_MODEL = 'no_movement'
STATES_FUNCTION = 'multi_agent'
MAX_STEPS = 40000
EVAL_STEPS = 1000
# MAX_STEPS = 1000
STEPS_PER_EPISODE = 5
EVAL_STEPS_PER_EPISODE = 10
REPLAY_INITIAL = 0
TEST_NUM_EPISODES = 10000
REPLAY_MEMORY_SIZE = int(10E3)
LEARNING_RATE = 8E-4
HIDDEN_SIZE = 64
N_HIDDEN_LAYERS = 1
BATCH_SIZE = 128
GAMMA = .5
EPSILON_INITIAL = 1
EPSILON_MIN = .01
EPSILON_DECAY = 1 / (.1 * MAX_STEPS)
SOFT_TAU = .05
ALPHA = .6
BETA = .4
REPLAY_MEMORY_TYPE = 'standard'
PRIO_BETA_ITS = int(.4*(MAX_STEPS - REPLAY_INITIAL))
PRINT_EVERY = int(MAX_STEPS/100)
EVAL_EVERY = int(MAX_STEPS / 20)
TARGET_UPDATE = 50
# EVAL_EVERY = int(MAX_STEPS / 1)
OU_DECAY_PERIOD = 100000
# OU_DECAY_PERIOD = STEPS_PER_EPISODE
# ou noise params
OU_MU = 0.2
OU_THETA = .25
OU_MAX_SIGMA = .5
OU_MIN_SIGMA = .05
OU_DIM = 1e-2
# adaptive noise params
INITIAL_STDDEV = 0.1
DESIRED_ACTION_STDDEV = 0.3
ADAPTATION_COEFFICIENT = 1.01
UPDATE_PERTURBERD_EVERY = 5
# normal actions noise
NORMAL_LOC = 0
NORMAL_SCALE = 4
NORMAL_T = MAX_STEPS
NORMAL_MIN_SCALE = .01

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
ref_env = CompleteEnvironment12(
    env_params,
    channel_to_bs,
    channel_to_devices,
    reward_penalty=REWARD_PENALTY,
    memory=ENVIRONMENT_MEMORY,
    bs_height=bs_height,
    reward_function=REWARD_FUNCTION,
    memories_capacity=int(1e4),
    dt=DELTA_T,
    states_function=STATES_FUNCTION
)
env = deepcopy(ref_env)
a_min = -60
a_offset = -10
# a_min = 0 + 1e-9
action_size = MAX_NUMBER_OF_AGENTS
env_state_size = ref_env.state_size()
# actions = db_five(p_min, p_max)
actions = db_six(p_min, p_max)
# actions = db_ten(p_min, p_max)
NUMBER_OF_ACTIONS = len(actions)
agent_params = DQNAgentParameters(
    EPSILON_MIN, EPSILON_DECAY, EPSILON_INITIAL, REPLAY_MEMORY_SIZE,
    BATCH_SIZE, GAMMA
)
param_noise = AdaptiveParamNoiseSpec(
    initial_stddev=INITIAL_STDDEV,
    desired_action_stddev=DESIRED_ACTION_STDDEV,
    adaptation_coefficient=ADAPTATION_COEFFICIENT
)
agent_params = DQNAgentParameters(
    EPSILON_MIN, EPSILON_DECAY, EPSILON_INITIAL, REPLAY_MEMORY_SIZE,
    BATCH_SIZE, GAMMA
)
agents = [ExternalDQNAgent(agent_params, actions)
          for _ in range(MAX_NUMBER_OF_AGENTS)]
action_range = range(len(actions))


def act():
    action_index = random.choice(action_range)
    action = actions[action_index]
    return action, action_index


def test():
    mue_availability = []
    mue_sinrs = []
    d2d_sinrs = []
    rewards_bag = []
    for _ in range(TEST_NUM_EPISODES):
        env.reset()
        env.build_scenario(agents, motion_model=MOTION_MODEL)
        obs, _, _, _ = env.step(agents)
        i = 0
        done = False
        ep_availability = []
        ep_rewards = []
        ep_mue_sinrs = []
        ep_d2d_sinrs = []
        while not done and i < EVAL_STEPS_PER_EPISODE:
            with torch.no_grad():
                for agent in agents:
                    action, a_index = act()
                    agent.action = action
                    agent.action_index = a_index
            # actions = np.zeros(MAX_NUMBER_OF_AGENTS) + 1e-9
            # db_actions = power_to_db(actions)
            next_obs, reward, done, _ = env.step(agents)
            obs = next_obs
            ep_availability.append(env.mue.sinr > env.params.sinr_threshold)
            ep_rewards.append(reward)
            ep_mue_sinrs.append(env.mue.sinr)
            ep_d2d_sinrs.append([p[0].sinr for p in env.d2d_pairs])
            i += 1
        rewards_bag += ep_rewards
        mue_sinrs += ep_mue_sinrs
        d2d_sinrs += ep_d2d_sinrs
        mue_availability += ep_availability
    all_bags = {
        'rewards': rewards_bag,
        'mue_sinrs': mue_sinrs,
        'd2d_sinrs': d2d_sinrs,
        'mue_availability': mue_availability
    }
    return all_bags


def run():
    # make data dir
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    dir_path = f'data/{ALGO_NAME}/{filename}'
    data_path, _ = gen.make_dir_timestamp(dir_path)
    start = time()
    # set environment up
    test_bags = test()
    # save stuff
    now = (time() - start) / 60
    data_file_path = f'{data_path}/log.pickle'
    data = {
        'test_bags': test_bags,
        'elapsed_time': now,
        'eval_every': EVAL_EVERY,
        'mue_sinr_threshold': sinr_threshold_train,
    }
    env.reset()
    save_with_pickle(env, f'{data_path}/env.pickle')
    save_with_pickle(data, data_file_path)
    copyfile(__file__, f'{data_path}/{filename}.py')
    print(f'done. Elapsed time: {now} minutes.')


if __name__ == '__main__':
    run()
