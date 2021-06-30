# Simulates a fixed amount of d2d pairs many times and calculates
# their spectral efficiencies. In the first scenario, the d2d
# powers are chosen randomly. In the second scenario, the d2d powers
# are always the same, just like what the AIs have been doing.
import sys_simulator.general as gen
import os
from sys_simulator.q_learning import rewards
from sys_simulator.parameters.parameters import EnvironmentParameters
from typing import List
import numpy as np
import pickle
import time
from sys_simulator.q_learning.environments.completeEnvironment2 \
    import CompleteEnvironment2
from sys_simulator.a2c.agent import Agent
import random


def test_random(env: CompleteEnvironment2, n_agents: int,
                num_episodes: int, episode_steps: int,
                actions: List[float]):
    """
    in this test, the agents choose their actions randomly,
    by selecting a random index in `actions_indexes`
    """
    mue_spectral_effs = []
    d2d_spectral_effs = []
    sinr_d2ds = []
    done = False
    bag = list()
    actions_indexes = [i for i in range(len(actions))]
    start = time.time()
    for ep in range(num_episodes):
        agents = [Agent() for _ in range(n_agents)]
        env.build_scenario(agents)
        done = False
        i = 0
        while not done and i < episode_steps:
            for _, agent in enumerate(agents):
                action_index = random.choice(actions_indexes)
                bag.append(action_index)
                agent.action_index = action_index
                agent.action = actions[action_index]
            _, _, done = env.step(agents)
            i += 1
        mue_spectral_effs.append(env.mue_spectral_eff.item())
        d2d_spectral_effs.append(env.d2d_spectral_eff.item())
        sinr_d2ds += env.sinr_d2ds
        now = (time.time() - start)/60
        print(f'Episode {ep}. Elapsed time: {now} minutes.')
        # action_counts[n_agents].append(gen.action_counts(env.sinr_d2ds))
    return mue_spectral_effs, d2d_spectral_effs, sinr_d2ds, bag


def test(env: CompleteEnvironment2, n_agents: int,
         num_episodes: int, episode_steps: int,
         actions, action_index: int):
    """
    in this tests, the agents choose the same action, which
    is defined by `action_index`, which statys constant throghout
    the whole test
    """
    mue_spectral_effs = []
    d2d_spectral_effs = []
    sinr_d2ds = []
    done = False
    bag = list()
    start = time.time()
    for ep in range(num_episodes):
        agents = [Agent() for _ in range(n_agents)]
        env.build_scenario(agents)
        done = False
        i = 0
        while not done and i < episode_steps:
            for _, agent in enumerate(agents):
                bag.append(action_index)
                agent.action_index = action_index
                agent.action = actions[action_index]
            _, _, done = env.step(agents)
            i += 1
        mue_spectral_effs.append(env.mue_spectral_eff.item())
        d2d_spectral_effs.append(env.d2d_spectral_eff.item())
        sinr_d2ds += env.sinr_d2ds
        now = (time.time() - start)/60
        print(f'Episode {ep}. Elapsed time: {now} minutes.')
        # action_counts[n_agents].append(gen.action_counts(env.sinr_d2ds))
    return mue_spectral_effs, d2d_spectral_effs, sinr_d2ds, bag


def run():
    n_mues = 1  # number of mues
    N_D2D = 2  # number of d2d pairs
    n_rb = n_mues   # number of RBs
    bs_radius = 500  # bs radius in m
    rb_bandwidth = 180*1e3  # rb bandwidth in Hz
    d2d_pair_distance = 50  # d2d pair distance in m
    p_max = 23  # max tx power in dBm
    noise_power = -116  # noise power per RB in dBm
    bs_gain = 17    # macro bs antenna gain in dBi
    user_gain = 4   # user antenna gain in dBi
    sinr_threshold_mue = 6  # true mue sinr threshold in dB
    mue_margin = .5e4
    # conversions from dB to pow
    p_max = p_max - 30
    p_max = gen.db_to_power(p_max)
    noise_power = noise_power - 30
    noise_power = gen.db_to_power(noise_power)
    bs_gain = gen.db_to_power(bs_gain)
    user_gain = gen.db_to_power(user_gain)
    sinr_threshold_mue = gen.db_to_power(sinr_threshold_mue)
    # q-learning parameters
    STEPS_PER_EPISODE = 1
    MAX_NUM_EPISODES = 2000
    C = 80  # C constant for the improved reward function
    NUM_ACTIONS = 5
    # more parameters
    cwd = os.getcwd()
    env_params = EnvironmentParameters(
        rb_bandwidth, d2d_pair_distance, p_max, noise_power,
        bs_gain, user_gain, sinr_threshold_mue,
        n_mues, N_D2D, n_rb, bs_radius, c_param=C, mue_margin=mue_margin)
    # inits
    reward_function = rewards.dis_reward_tensor2
    environment = CompleteEnvironment2(env_params, reward_function)
    reward_function = rewards.dis_reward_tensor
    # actions
    actions = [i*0.82*p_max/5/1000 for i in range(NUM_ACTIONS)]  # best result
    # test func call
    # mue_spectral_effs, d2d_spectral_effs, sinr_d2ds, bag = \
    #     test_random(environment, N_D2D, MAX_NUM_EPISODES,
    #          STEPS_PER_EPISODE, actions)
    mue_spectral_effs, d2d_spectral_effs, sinr_d2ds, bag = \
        test(environment, N_D2D, MAX_NUM_EPISODES,
             STEPS_PER_EPISODE, actions, 3)
    # mue success rate
    mue_success_rate = list()
    for i, m in enumerate(mue_spectral_effs):
        mue_success_rate.append(
            np.average(m > np.log2(1 + sinr_threshold_mue)))
    # sum of d2d spectral efficiencies
    d2d_speffs_total = d2d_spectral_effs
    # d2d individual spectral efficiencies
    sinr_d2ds = np.array(sinr_d2ds)
    d2d_individual_speffs = np.log2(1 + sinr_d2ds)
    d2d_individual_mean_speffs = np.mean(d2d_individual_speffs, axis=1)
    # text log
    log = list()
    for i, d in enumerate(zip(d2d_speffs_total, mue_success_rate)):
        log.append(f'NUMBER OF D2D_USERS: {i+1}')
        log.append(f'D2D SPECTRAL EFFICIENCY - SCRIPT: {d[0]}')
        log.append(f'MUE SUCCESS RATE - SCRIPT: {d[1]}')
        log.append('-------------------------------------')
    # filename
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    pickle_filename = f'{cwd}/data/general/{filename}.pickle'
    # pickle data
    data = {
        'd2d_speffs_total': d2d_speffs_total,
        'mue_success_rate': mue_success_rate,
        'chosen_actions': bag,
        'd2d_individual_speffs': d2d_individual_speffs,
        'd2d_individual_mean_speffs': d2d_individual_mean_speffs
    }
    # writing pickle file
    with open(pickle_filename, 'wb') as file:
        pickle.dump(data, file)
    # text log
    filename = f'{cwd}/logs/general/{filename}.txt'
    file = open(filename, 'w')
    for lg in log:
        file.write(f'{lg}\n')
    file.close()
