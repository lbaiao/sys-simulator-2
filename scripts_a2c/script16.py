from copy import deepcopy
from math import ceil
from multiprocessing import Pool
from shutil import copyfile
from sys_simulator.a2c.parallel import env_step, env_step_test
from sys_simulator.a2c.agent import A2CAgent, A2CCentralAgent
from sys_simulator.a2c.framework import A2CDiscreteFramework
from sys_simulator.general.actions_discretizations import db_five, db_six, db_ten
from sys_simulator.dqn.externalDQNFramework import ExternalDQNFramework
from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from sys_simulator.noises.decaying_gauss_noise import DecayingGaussNoise
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
    print_evaluating,
    print_stuff_ddpg,
    random_seed,
    save_with_pickle,
)
from sys_simulator.noises.ou_noise import OUNoise2, SysSimOUNoise
from sys_simulator.parameters.parameters import DQNAgentParameters, EnvironmentParameters
from sys_simulator.plots import plot_env_states, plot_trajectories
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
DELTA_T = 1e-3
# q-learning parameters
# training
ALGO_NAME = 'a2c'
OPTIMIZERS = 'adam'
NUM_ENVS = 4
NUM_POOLS = 4
REWARD_FUNCTION = 'multi_agent_continuous'
STATES_OPTIONS = ['sinrs', 'positions', 'channels']
MOTION_MODEL = 'random'
STATES_FUNCTION = 'multi_agent'
# MAX_STEPS = 1000
MAX_STEPS = 40000
EVAL_STEPS = 1000
# MAX_STEPS = 1000
STEPS_PER_EPISODE = 5
EVAL_STEPS_PER_EPISODE = 10
TEST_NUM_EPISODES = ceil(1000 / NUM_POOLS)
REPLAY_MEMORY_SIZE = int(10E3)
ACTOR_LEARNING_RATE = 2E-4
CRITIC_LEARNING_RATE = 2E-3
HIDDEN_SIZE = 64
N_HIDDEN_LAYERS = 1
GAMMA = .5
LBDA = .95
BETA = 1e-2
PRINT_EVERY = int(MAX_STEPS/100)
EVAL_EVERY = int(MAX_STEPS/20)
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
    memories_capacity=int(5e2),
    dt=DELTA_T,
    states_function=STATES_FUNCTION
)
envs = [deepcopy(ref_env) for _ in range(NUM_ENVS)]
a_min = -60
a_max = 60
a_offset = -10
# a_min = 0 + 1e-9
# a_max = db_to_power(p_max - 10)
action_size = MAX_NUMBER_OF_AGENTS
env_state_size = ref_env.state_size()
# actions = db_five(p_min, p_max)
actions = db_six(p_min, p_max)
# actions = [-90, -60, -40, -30, -20, -10]
# actions = db_ten(p_min, p_max)
NUMBER_OF_ACTIONS = len(actions)
framework = A2CDiscreteFramework(
    env_state_size,
    len(actions),
    HIDDEN_SIZE,
    N_HIDDEN_LAYERS,
    STEPS_PER_EPISODE,
    NUM_ENVS*MAX_NUMBER_OF_AGENTS,
    ACTOR_LEARNING_RATE,
    CRITIC_LEARNING_RATE,
    BETA,
    GAMMA,
    LBDA,
    OPTIMIZERS,
    torch_device
)
best_framework = deepcopy(framework)
param_noise = AdaptiveParamNoiseSpec(
    initial_stddev=INITIAL_STDDEV,
    desired_action_stddev=DESIRED_ACTION_STDDEV,
    adaptation_coefficient=ADAPTATION_COEFFICIENT
)
ou_noise = SysSimOUNoise(
    action_size,
    a_min, a_max,
    OU_MU, OU_THETA,
    OU_MAX_SIGMA,
    OU_MIN_SIGMA,
    OU_DECAY_PERIOD
)
ou_noise2 = OUNoise2(
    OU_MU,
    action_size,
    OU_MAX_SIGMA,
    OU_THETA,
    OU_DIM
)
decaying_noise = DecayingGaussNoise(
    NORMAL_LOC, NORMAL_SCALE, NORMAL_T, NORMAL_MIN_SCALE, action_size
)
agents = [[
    A2CAgent(torch_device, actions)
    for _ in range(MAX_NUMBER_OF_AGENTS)
] for _ in range(NUM_ENVS)]
central_agent = A2CCentralAgent(torch_device)


def train(start: float, writer: SummaryWriter, pool):
    global envs
    global agents
    mue_sinrs_bag = list()
    d2d_sinrs_bag = list()
    rewards_bag = list()
    mue_avail_bag = list()
    step = 0
    collected_states = list()
    best_avg_reward = float('-inf')
    while step < MAX_STEPS:
        for e, a in zip(envs, agents):
            e.reset()
            e.build_scenario(a, d2d_limited_power=False,
                             motion_model=MOTION_MODEL)
        obs, _, _, envs, agents = env_step(pool, envs, agents)
        i = 0
        total_entropy = 0
        while i < STEPS_PER_EPISODE:
            action, log_prob, entropy, value = \
                central_agent.act(obs, framework)
            a_index = 0
            for ag in agents:
                for a in ag:
                    a.action_index = action[a_index].item()
                    a.action = actions[a.action_index]
                    a_index += 1
            next_obs, reward, done, envs, agents = env_step(pool, envs, agents)
            total_entropy += entropy
            framework.push_experience(log_prob, value,
                                      reward, done, obs, action)
            obs = next_obs
            i += 1
            step += 1
            if step % PRINT_EVERY == 0:
                now = (time() - start) / 60
                print_stuff_ddpg(step, now, MAX_STEPS, 'standard')
            # testing
            if step % EVAL_EVERY != 0:
                continue
            t_bags = test(framework, pool)
            t_rewards = t_bags['rewards']
            t_mue_sinrs = t_bags['mue_sinrs']
            t_d2d_sinrs = t_bags['d2d_sinrs']
            t_availability = t_bags['mue_availability']
            # mue spectral eff
            mue_sinrs_bag.append(t_mue_sinrs)
            # average d2d spectral eff
            d2d_sinrs_bag.append(t_d2d_sinrs)
            rewards_bag.append(t_rewards)
            mue_avail_bag.append(t_availability)
            # write metrics
            t_mue_sinrs = np.mean(t_mue_sinrs, axis=0)
            t_d2d_sinrs = np.mean(t_d2d_sinrs, axis=0)
            t_d2d_sinrs = {f'device {i}': a for i, a in enumerate(t_d2d_sinrs)}
            # t_d2d_sinrs = np.mean(t_d2d_sinrs)
            t_availability = np.mean(t_availability, axis=0)
            t_rewards = np.mean(t_rewards)
            if t_rewards > best_avg_reward:
                best_framework.a2c.actor.load_state_dict(
                    framework.a2c.actor.state_dict())
                best_avg_reward = t_rewards
            writer.add_scalar(
                '2. Testing - Average MUE SINRs [dB]',
                t_mue_sinrs,
                step
            )
            writer.add_scalars(
                '2. Testing - Average D2D SINRs [dB]',
                t_d2d_sinrs,
                step
            )
            writer.add_scalar(
                '2. Testing - Aggregated Rewards', t_rewards, step)
            writer.add_scalar('2. Testing - Average MUE Availability',
                              np.mean(t_availability), step)
        _, _, _, next_value = central_agent.act(next_obs, framework)
        framework.push_next(next_obs, next_value, total_entropy)
        a_loss, c_loss = framework.learn()
        writer.add_scalar('1. Training - Actor Losses', a_loss, step)
        writer.add_scalar('1. Training - Critic Losses', c_loss, step)
        # adaptive param noise
    all_bags = {
        'mue_sinrs': mue_sinrs_bag,
        'd2d_sinrs': d2d_sinrs_bag,
        'collected_states': collected_states,
        'rewards': rewards_bag,
    }
    return all_bags


def test(framework: A2CDiscreteFramework, pool):
    global envs
    global agents
    mue_availability = []
    mue_sinrs = []
    d2d_sinrs = []
    rewards_bag = []
    for _ in range(TEST_NUM_EPISODES):
        for e, a in zip(envs, agents):
            e.reset()
            e.build_scenario(a, d2d_limited_power=False,
                             motion_model=MOTION_MODEL)
        obs, _, _, envs, agents = env_step(pool, envs, agents)
        i = 0
        done = False
        ep_availability = []
        ep_rewards = []
        ep_mue_sinrs = []
        ep_d2d_sinrs = []
        while i < EVAL_STEPS_PER_EPISODE:
            action, log_prob, entropy, value = \
                central_agent.act(obs, framework)
            a_index = 0
            for ag in agents:
                for a in ag:
                    a.action_index = action[a_index].item()
                    a.action = actions[a.action_index]
                    a_index += 1
            next_obs, reward, done, envs, agents = \
                env_step_test(pool, envs, agents)
            obs = next_obs
            st_mue_sinrs = np.array([e.mue.sinr for e in envs])
            avails = st_mue_sinrs > envs[0].params.sinr_threshold
            env_pairs = [p for p in [e.d2d_pairs for e in envs]]
            st_d2d_sinrs = []
            for ep in env_pairs:
                sinrs = [p[0].sinr for p in ep]
                st_d2d_sinrs.append(sinrs)
            ep_availability += avails.tolist()
            ep_rewards += reward.tolist()
            ep_mue_sinrs += [*st_mue_sinrs]
            ep_d2d_sinrs += st_d2d_sinrs
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


def evaluate(start: float, writer: SummaryWriter):
    step = 0
    mue_availability = []
    env.reset()
    env.build_scenario(agents, motion_model=MOTION_MODEL)
    env.reset_devices_positions()
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
                agent.get_action(framework, obs[j])
                past_actions[j] = agent.action_index
        next_obs, reward, _, _ = env.step(agents)
        obs = next_obs
        i += 1
        step += 1
        rewards = {f'device {i}': r for i, r in enumerate(reward)}
        writer.add_scalars('3. Eval - Rewards', rewards, step)
        sinrs = [a.d2d_tx.sinr for a in agents]
        d2d_sinrs.append(sinrs)
        sinrs = {f'device {i}': s for i, s in enumerate(sinrs)}
        writer.add_scalars('3. Eval - SINRs [dB]', sinrs, step)
        writer.add_scalars(
            '3. Eval - Transmission powers [dBW]',
            {f'device {i}': a for i, a in enumerate(past_actions)},
            step
        )
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


def run():
    # multiprocessing pool
    pool = Pool(NUM_POOLS)
    # make data dir
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    dir_path = f'data/{ALGO_NAME}/{filename}'
    data_path, _ = gen.make_dir_timestamp(dir_path)
    writer = SummaryWriter(f'{data_path}/tensorboard')
    start = time()
    # set environment up
    train_bags = train(start, writer, pool)
    # states = train_bags['collected_states']
    # plot_env_states(states, N_STATES_BINS, f'{data_path}/env_states.png')
    test_bags = test(framework, pool)
    # evaluate(start, writer)
    writer.close()
    # save stuff
    now = (time() - start) / 60
    data_file_path = f'{data_path}/log.pickle'
    data = {
        'train_bags': train_bags,
        'test_bags': test_bags,
        'elapsed_time': now,
        'eval_every': EVAL_EVERY,
        'mue_sinr_threshold': sinr_threshold_train,
    }
    save_with_pickle(envs[0], f'{data_path}/env.pickle')
    save_with_pickle(data, data_file_path)
    copyfile(__file__, f'{data_path}/{filename}.py')
    torch.save(framework, f'{data_path}/last_model.pt')
    torch.save(best_framework, f'{data_path}/best_model.pt')
    print(f'done. Elapsed time: {now} minutes.')
    pool.close()


if __name__ == '__main__':
    run()
