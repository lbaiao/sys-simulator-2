#  Testing for a3c/script7


import sys_simulator.general.general as gen
import os
from sys_simulator.q_learning import rewards
from sys_simulator.parameters.parameters import EnvironmentParameters
from typing import List
import torch
import numpy as np
import pickle
import time
from sys_simulator.q_learning.environments.completeEnvironmentA2C2 \
    import CompleteEnvironmentA2C2
from sys_simulator.a2c.agent import Agent
from sys_simulator.a2c.a2c import ActorCriticDiscrete


def test(env: CompleteEnvironmentA2C2, framework: ActorCriticDiscrete,
         max_d2d: int, num_episodes: int, episode_steps: int,
         aux_range: List[int], actions: List[float]):
    mue_spectral_effs = [list() for i in range(max_d2d+1)]
    d2d_spectral_effs = [list() for i in range(max_d2d+1)]
    done = False
    bag = list()
    # aux_range = range(max_d2d+1)[1:]
    start = time.time()
    for ep in range(num_episodes):
        n_agents = np.random.choice(aux_range)
        agents = [Agent() for _ in range(n_agents)]
        env.build_scenario(agents)
        done = False
        obs = [env.get_state(a) for a in agents]
        i = 0
        while not done and i < episode_steps:
            for j, agent in enumerate(agents):
                action_index, _, _ = agent.act_discrete(framework, obs[j])
                bag.append(action_index.item())
                agent.set_action(actions[action_index.item()])
            next_obs, _, done = env.step(agents)
            obs = next_obs
            i += 1
        mue_spectral_effs[n_agents].append(env.mue_spectral_eff.item())
        d2d_spectral_effs[n_agents].append(env.d2d_spectral_eff.item())
        now = (time.time() - start)/60
        print(f'Episode {ep}. Elapsed time: {now} minutes.')
        # action_counts[n_agents].append(gen.action_counts(env.sinr_d2ds))
    return mue_spectral_effs, d2d_spectral_effs, bag


def run():
    n_mues = 1  # number of mues
    n_d2d = 2  # number of d2d pairs
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
    STEPS_PER_EPISODE = 20
    MAX_NUM_EPISODES = 2000
    # MAX_NUM_EPISODES = 2
    C = 80  # C constant for the improved reward function
    MAX_NUMBER_OF_AGENTS = 10
    NUM_ACTIONS = 5
    HIDDEN_SIZE = 256
    # mu = 0.82*p_max/5/2000
    # std = mu/6
    mu = p_max*1e-8
    std = mu/100
    # more parameters
    cwd = os.getcwd()
    env_params = EnvironmentParameters(
        rb_bandwidth, d2d_pair_distance, p_max, noise_power,
        bs_gain, user_gain, sinr_threshold_mue,
        n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin)
    reward_function = rewards.dis_reward_tensor2
    environment = CompleteEnvironmentA2C2(env_params, reward_function)

    framework = ActorCriticDiscrete(environment.state_space_size,
                                    NUM_ACTIONS, HIDDEN_SIZE, mu, std)
    framework.load_state_dict(torch.load(f'{cwd}/models/a3c/script7.pt'))

    reward_function = rewards.dis_reward_tensor

    # policy 5 test
    aux_range = list(range(MAX_NUMBER_OF_AGENTS+1))[1:]
    actions = np.linspace(4.5e-5, 6e-5, 5)[::-1] * p_max
    actions[0] = 0.0
    mue_spectral_effs, d2d_spectral_effs, bag, = \
        test(environment, framework, MAX_NUMBER_OF_AGENTS, MAX_NUM_EPISODES,
             STEPS_PER_EPISODE, aux_range, actions)

    mue_success_rate = list()
    for i, m in enumerate(mue_spectral_effs):
        mue_success_rate.append(
            np.average(m > np.log2(1 + sinr_threshold_mue)))

    d2d_speffs_avg = list()
    for i, d in enumerate(d2d_spectral_effs):
        d2d_speffs_avg.append(np.average(d))

    log = list()
    for i, d in enumerate(zip(d2d_speffs_avg, mue_success_rate)):
        log.append(f'NUMBER OF D2D_USERS: {i+1}')
        log.append(f'D2D SPECTRAL EFFICIENCY - SCRIPT: {d[0]}')
        log.append(f'MUE SUCCESS RATE - SCRIPT: {d[1]}')
        log.append('-------------------------------------')

    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]

    pickle_filename = f'{cwd}/data/a3c/{filename}.pickle'

    data = {
        'd2d_speffs_avg_total': d2d_spectral_effs,
        'mue_success_rate': mue_success_rate,
        'chosen_actions': bag,
    }

    with open(pickle_filename, 'wb') as file:
        pickle.dump(data, file)

    filename = f'{cwd}/logs/a3c/{filename}.txt'
    file = open(filename, 'w')

    for lg in log:
        file.write(f'{lg}\n')
    file.close()
