from copy import deepcopy
from shutil import copyfile
from sys_simulator.noises.decaying_gauss_noise import DecayingGaussNoise
from time import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sys_simulator.channels import BANChannel, UrbanMacroNLOSWinnerChannel
from sys_simulator.ddpg.agent import SurrogateAgent, SysSimAgent, SysSimAgentWriter
from sys_simulator.ddpg.framework import Framework, PerturberdFramework
from sys_simulator.ddpg.parameters_noise import (
    AdaptiveParamNoiseSpec,
    ddpg_distance_metric,
)
from sys_simulator.devices.devices import db_to_power
import sys_simulator.general as gen
from sys_simulator.general import (
    power_to_db,
    print_evaluating,
    print_stuff_ddpg,
    random_seed,
)
from sys_simulator.noises.ou_noise import OUNoise2, SysSimOUNoise
from sys_simulator.parameters.parameters import EnvironmentParameters
from sys_simulator.plots import plot_env_states, plot_positions, plot_trajectories
from sys_simulator.q_learning.environments.completeEnvironment12 import (
    CompleteEnvironment12,
)

# Uses CompleteEnvironment12
# Centralized Learning-Centralized Execution
# NN Parameters noise.
# Similar to script5.
# Trains a central agent for a fixed amount of devices.
# There are different channels to the BS and to the devices.
# The script generates a single scenario and tries to find
# the optimal solution for it.
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
mue_margin = 20  # mue margin in dB
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
N_STATES_BINS = 100
DELTA_T = 1e-3
# q-learning parameters
# training
ALGO_NAME = 'ddpg'
REWARD_FUNCTION = 'jain'
STATES_OPTIONS = ['sinrs', 'positions', 'channels']
MOTION_MODEL = 'random'
# MAX_STEPS = 1000
MAX_STEPS = 6000
EVAL_STEPS = 2000
# MAX_STEPS = 1000
STEPS_PER_EPISODE = 100
EVAL_STEPS_PER_EPISODE = 50
REPLAY_INITIAL = 0
TEST_NUM_EPISODES = 5
REPLAY_MEMORY_SIZE = int(10E3)
ACTOR_LEARNING_RATE = 2E-4
CRITIC_LEARNING_RATE = 2E-3
HIDDEN_SIZE = 64
N_HIDDEN_LAYERS = 0
BATCH_SIZE = 64
GAMMA = .99
SOFT_TAU = .05
ALPHA = .6
BETA = .4
EXPLORATION = 'perturberd'
REPLAY_MEMORY_TYPE = 'standard'
PRIO_BETA_ITS = int(.8*(MAX_STEPS - REPLAY_INITIAL))
PRINT_EVERY = int(MAX_STEPS/100)
EVAL_EVERY = int(MAX_STEPS / 10)
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
    memories_capacity=int(5e2),
    dt=DELTA_T
)
a_min = -60
a_max = 60
a_offset = -10
# a_min = 0 + 1e-9
# a_max = db_to_power(p_max - 10)
action_size = MAX_NUMBER_OF_AGENTS
env_state_size = ref_env.state_size()

framework = Framework(
    REPLAY_MEMORY_TYPE,
    REPLAY_MEMORY_SIZE,
    REPLAY_INITIAL,
    env_state_size,
    action_size,
    HIDDEN_SIZE,
    N_HIDDEN_LAYERS,
    ACTOR_LEARNING_RATE,
    CRITIC_LEARNING_RATE,
    BATCH_SIZE,
    GAMMA,
    SOFT_TAU,
    torch_device,
    alpha=ALPHA,
    beta=BETA,
    beta_its=PRIO_BETA_ITS
)
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


central_agent = SysSimAgentWriter(a_min, a_max, EXPLORATION,
                                  torch_device, a_offset=a_offset)
central_agent_test = SysSimAgent(a_min, a_max, EXPLORATION,
                                 torch_device, a_offset=a_offset)
surr_agents = [SurrogateAgent() for _ in range(MAX_NUMBER_OF_AGENTS)]


def train(env: CompleteEnvironment12, start: float, writer: SummaryWriter):
    actor_losses_bag = list()
    critic_losses_bag = list()
    mue_spectral_eff_bag = list()
    d2d_spectral_eff_bag = list()
    rewards_bag = list()
    mue_avail_bag = list()
    step = 0
    collected_states = list()
    while step < MAX_STEPS:
        env.reset_devices_positions()
        obs, _, _, _ = env.step(surr_agents)
        total_reward = 0.0
        done = False
        i = 0
        framework.perturb_actor_parameters(param_noise)
        while not done and i < STEPS_PER_EPISODE:
            if step < REPLAY_INITIAL:
                actions = 2 * \
                    (np.random.random(MAX_NUMBER_OF_AGENTS) + 1e-9 - .5)
                if np.sum(actions == 0) > 0:
                    actions += 1e-9
            else:
                ou_noise2.reset()
                actions = central_agent.act(
                    obs, framework,
                    writer, step, True,
                    noise=np.random.normal(loc=NORMAL_LOC, scale=NORMAL_SCALE),
                    param_noise=param_noise
                )
            # db_actions = power_to_db(actions)
            db_actions = actions
            for j, agent in enumerate(surr_agents):
                agent.set_action(db_actions[j].item())
            next_obs, rewards, done, _ = env.step(surr_agents)
            collected_states.append(next_obs)
            total_reward = np.sum(rewards)
            framework.replay_memory.push(
                obs, actions,
                total_reward, next_obs, done
            )
            actor_loss, critic_loss = framework.learn()
            obs = next_obs
            i += 1
            step += 1
            d2d_powers = [p[0].tx_power for p in env.d2d_pairs]
            writer.add_scalar('1. Training - Actor Losses', actor_loss, step)
            writer.add_scalar('1. Training - Critic Losses', critic_loss, step)
            writer.add_scalars(
                '1. Training - D2D powers [dBW]',
                {f'device {i}': a for i, a in enumerate(d2d_powers)},
                step
            )
            writer.add_scalar(
                '1. Training - MUE SINR [dB]', env.mue.sinr, step)
            writer.add_scalar('1. Training - MUE Tx Power [dB]',
                              env.mue.tx_power, step)
            mue_bs_loss = env.total_losses[env.mue.id][env.bs.id]
            writer.add_scalar('1. Training - Channel loss MUE to BS [dB]',
                              mue_bs_loss, step)
            # writer.add_scalar('Aggregated Actions', aux_actions.sum(), step)
            actor_losses_bag.append(actor_loss)
            critic_losses_bag.append(critic_loss)
            framework.actor.train()
            framework.critic.train()
            if step % PRINT_EVERY == 0:
                now = (time() - start) / 60
                print_stuff_ddpg(step, now, MAX_STEPS, REPLAY_MEMORY_TYPE)
            if step % UPDATE_PERTURBERD_EVERY == 0:
                framework.perturb_actor_parameters(param_noise)
            # testing
            if step % EVAL_EVERY != 0:
                continue
            t_bags = test(env, framework)
            t_rewards = t_bags['rewards']
            t_mue_sinrs = t_bags['mue_sinrs']
            t_d2d_sinrs = t_bags['d2d_sinrs']
            t_availability = t_bags['mue_availability']
            # mue spectral eff
            mue_spectral_eff_bag.append(t_mue_sinrs)
            # average d2d spectral eff
            d2d_spectral_eff_bag.append(t_d2d_sinrs)
            rewards_bag.append(t_rewards)
            mue_avail_bag.append(t_availability)
            # write metrics
            t_mue_sinrs = np.mean(t_mue_sinrs, axis=0)
            t_d2d_sinrs = np.mean(t_d2d_sinrs, axis=0)
            t_d2d_sinrs = {f'device {i}': a for i, a in enumerate(t_d2d_sinrs)}
            # t_d2d_sinrs = np.mean(t_d2d_sinrs)
            t_availability = np.mean(t_availability, axis=0)
            t_rewards = np.mean(t_rewards)
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
        # adaptive param noise
        memory = deepcopy(framework.replay_memory)
        if memory._next_idx - i > 0:
            noise_data = \
                memory._storage[memory._next_idx - i: memory._next_idx]
        else:
            noise_data = \
                memory._storage[
                    memory._next_idx - i +
                    REPLAY_MEMORY_SIZE: REPLAY_MEMORY_SIZE
                ] + memory._storage[0: memory._next_idx]
        noise_data = np.array(noise_data)
        noise_s, perturbed_actions, _, _, _ = zip(*noise_data)
        noise_s = np.array(noise_s)
        noise_s = torch.FloatTensor(noise_s).to(torch_device)
        unperturbed_actions = framework.actor(noise_s)
        unperturbed_actions = unperturbed_actions.detach().cpu().numpy()
        ddpg_dist = ddpg_distance_metric(
            perturbed_actions, unperturbed_actions)
        param_noise.adapt(ddpg_dist)
    all_bags = {
        'actor_losses': actor_losses_bag,
        'critic_losses': critic_losses_bag,
        'mue_spectral_effs': mue_spectral_eff_bag,
        'd2d_spectral_effs': d2d_spectral_eff_bag,
        'collected_states': collected_states
    }
    return all_bags, env


def test(env: CompleteEnvironment12, framework: Framework):
    framework.actor.eval()
    framework.critic.eval()
    mue_availability = []
    mue_sinrs = []
    d2d_sinrs = []
    rewards_bag = []
    for _ in range(TEST_NUM_EPISODES):
        env.reset_devices_positions()
        obs, _, _, _ = env.step(surr_agents)
        i = 0
        done = False
        ep_availability = []
        ep_rewards = []
        ep_mue_sinrs = []
        ep_d2d_sinrs = []
        while not done and i < EVAL_STEPS_PER_EPISODE:
            actions = central_agent_test.act(obs, framework, False)
            # actions = np.zeros(MAX_NUMBER_OF_AGENTS) + 1e-9
            # db_actions = power_to_db(actions)
            db_actions = actions
            for j, agent in enumerate(surr_agents):
                agent.set_action(db_actions[j].item())
            next_obs, reward, done, _ = env.step(surr_agents)
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


def evaluate(start: float, writer: SummaryWriter, env: CompleteEnvironment12):
    step = 0
    mue_availability = []
    env.reset_devices_positions()
    devices = env.get_devices()
    trajectories = {d.id: [d.position] for d in devices}
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
        sinrs = {f'device {i}': s for i, s in enumerate(sinrs)}
        writer.add_scalars('3. Eval - SINRs [dB]', sinrs, step)
        writer.add_scalars(
            '3. Eval - Transmission powers [dBW]',
            {f'device {i}': a for i, a in enumerate(db_actions)},
            step
        )
        mue_success = int(env.mue.sinr > env.params.sinr_threshold)
        mue_availability.append(mue_success)
        writer.add_scalar('3. Eval - MUE success', mue_success, step)
        for d in env.get_devices():
            trajectories[d.id].append(d.position)
    mue_availability = np.mean(mue_availability)
    writer.add_text('3. Eval - Average MUE availability',
                    str(mue_availability), step)
    writer.add_figure('3. Eval - Trajectories',
                      plot_trajectories(env, trajectories), step)


def run():
    # make data dir
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    dir_path = f'data/{ALGO_NAME}/{filename}'
    data_path, _ = gen.make_dir_timestamp(dir_path)
    writer = SummaryWriter(f'{data_path}/tensorboard')
    start = time()
    # set environment up
    env = deepcopy(ref_env)
    env.build_scenario(surr_agents, d2d_limited_power=False, 
                       motion_model=MOTION_MODEL)
    positions_fig = plot_positions(
        env.bs, [env.mue],
        [p[0] for p in env.d2d_pairs],
        [p[1] for p in env.d2d_pairs],
        False
    )
    writer.add_figure('Devices positions', positions_fig)
    train_bags, env = train(env, start, writer)
    states = train_bags['collected_states']
    plot_env_states(states, N_STATES_BINS, f'{data_path}/env_states.png')
    test_bags = test(env, framework)
    evaluate(start, writer, env)
    writer.close()
    # save stuff
    now = (time() - start) / 60
    data_file_path = f'{data_path}/log.pickle'
    data = {
        'train_bags': train_bags,
        'test_bags': test_bags,
        'elapsed_time': now,
        'eval_every': EVAL_EVERY,
    }
    gen.save_with_pickle(data, data_file_path)
    copyfile(__file__, f'{data_path}/{filename}.py')
    print(f'done. Elapsed time: {now} minutes.')


if __name__ == '__main__':
    run()
