# Uses CompleteEnvironment12
# Choose the power levels according to an uniform distribution.
# Benchmark for discrete power levels.
# Trains a central agent for a fixed amount of devices.
# There are different channels to the BS and to the devices.
# There are multiple episodes, where the devices are distributed
# in different positions, and there are different motion models.
from copy import deepcopy
from random import uniform
from shutil import copyfile
from typing import List
from sys_simulator.general.actions_discretizations import db_six
from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from time import time

import torch

from sys_simulator.channels import BANChannel, UrbanMacroNLOSWinnerChannel
import sys_simulator.general as gen
from sys_simulator.general import (
    db_to_power,
    power_to_db,
    print_evaluate3,
    random_seed,
    save_with_pickle,
)
from sys_simulator.parameters.parameters import DQNAgentParameters, EnvironmentParameters
from sys_simulator.q_learning.environments.completeEnvironment12 import (
    CompleteEnvironment12,
)

# training
ALGO_NAME = 'benchmark'
EVAL_STEPS_PER_EPISODE = 10
TEST_NUM_EPISODES = 1000
PRINT_EVERY = 100
AGENTS_RANGE = range(6)[1:]
# env params
REWARD_FUNCTION = 'simple'
STATES_OPTIONS = []
MOTION_MODEL = 'random'
STATES_FUNCTION = 'multi_agent'
JAIN_REWARD_PARAMETERS = {
    'gamma1': 10.0,
    'gamma2': 1.0,
    'gamma3': 0.0
}
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
    memories_capacity=int(1e4),
    dt=DELTA_T,
    states_function=STATES_FUNCTION,
    rewards_params=JAIN_REWARD_PARAMETERS
)
env = deepcopy(ref_env)
a_min = -60
a_offset = -10
action_size = MAX_NUMBER_OF_AGENTS
env_state_size = ref_env.state_size()
actions = db_six(p_min, p_max)
NUMBER_OF_ACTIONS = len(actions)
agent_params = DQNAgentParameters(1, 1, 1, 1, 1, 1)
agents = [ExternalDQNAgent(agent_params, actions)
          for _ in range(MAX_NUMBER_OF_AGENTS)]
action_range = range(len(actions))
p_max_lin = db_to_power(p_max)


def act():
    action = uniform(-90, p_max)
    action = power_to_db(action)
    return action


def test(start: float, agents: List[ExternalDQNAgent]):
    mue_availability = []
    mue_sinrs = []
    d2d_sinrs = []
    rewards_bag = []
    for ep in range(TEST_NUM_EPISODES):
        env = deepcopy(ref_env)
        env.build_scenario(agents, motion_model=MOTION_MODEL)
        env.step(agents)
        i = 0
        done = False
        ep_availability = []
        ep_rewards = []
        ep_mue_sinrs = []
        ep_d2d_sinrs = []
        if ep % PRINT_EVERY == 0:
            now = (time() - start) / 60
            print_evaluate3(ep, TEST_NUM_EPISODES, now, len(agents))
        while not done and i < EVAL_STEPS_PER_EPISODE:
            for agent in agents:
                action = act()
                agent.action = action
            # actions = np.zeros(MAX_NUMBER_OF_AGENTS) + 1e-9
            # db_actions = power_to_db(actions)
            _, reward, done, _ = env.step(agents)
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


if __name__ == '__main__':
    # make data dir
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    dir_path = f'data/{ALGO_NAME}/{filename}'
    data_path, _ = gen.make_dir_timestamp(dir_path)
    start = time()
    results = []
    for i in AGENTS_RANGE:
        agents = [
            ExternalDQNAgent(agent_params, actions)
            for _ in range(i)
        ]
        r = test(start, agents)
        results.append(r)
    # save stuff
    now = (time() - start) / 60
    data_file_path = f'{data_path}/log.pickle'
    save_with_pickle(results, data_file_path)
    copyfile(__file__, f'{data_path}/{filename}.py')
    print(f'done. Elapsed time: {now} minutes.')
