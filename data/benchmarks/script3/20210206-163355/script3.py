# Uses CompleteEnvironment10dB
# Random search for the best power
# Simulates only one episode and extracts the rewards, the pathloss_to_bs
# Simulates many times, for different number of agents, and take the averages.
# There are different channels to the BS and to the devices.
import random
from shutil import copyfile
from itertools import product
from time import time
from typing import List
from sys_simulator.general \
    import db_to_power, make_dir_timestamp, power_to_db, save_with_pickle
from sys_simulator.channels import BANChannel, UrbanMacroNLOSWinnerChannel
from sys_simulator import general as gen
from sys_simulator.q_learning.environments.completeEnvironment10dB \
    import CompleteEnvironment10dB
from sys_simulator.dqn.agents.dqnAgent import CentralDQNAgent, ExternalDQNAgent
from sys_simulator.dqn.externalDQNFramework import ExternalDQNFramework
from sys_simulator.parameters.parameters import \
    EnvironmentParameters, TrainingParameters, DQNAgentParameters
from sys_simulator.q_learning.rewards import dis_reward_tensor_db
from copy import deepcopy
import torch
import numpy as np


n_mues = 1  # number of mues
n_d2d = 2  # number of d2d pairs
n_rb = n_mues   # number of RBs
carrier_frequency = 2.4  # carrier frequency in GHz
bs_radius = 500  # bs radius in m
rb_bandwidth = 180*1e3  # rb bandwidth in Hz
d2d_pair_distance = 50  # d2d pair distance in m
device_height = 1.5  # mobile devices height in m
bs_height = 25  # BS antenna height in m
p_max = 40  # max tx power in dBm
noise_power = -116  # noise power per RB in dBm
bs_gain = 17    # macro bs antenna gain in dBi
user_gain = 4   # user antenna gain in dBi
sinr_threshold_train = 6  # mue sinr threshold in dB for training
mue_margin = 200  # mue margin in dB
# conversions from dBm to dB
p_max = p_max - 30
noise_power = noise_power - 30
# channel parameters
CHANNEL_RND = True
# q-learning parameters
# training
NUMBER = 1
STEPS_PER_EPISODE = 1000
TEST_STEPS_PER_EPISODE = 100
# STEPS_PER_EPISODE = 2
# TEST_STEPS_PER_EPISODE = 2
# common
EPSILON_INITIAL = 1
EPSILON_MIN = .05
EPSILON_DECAY = 1.3 / STEPS_PER_EPISODE    # fast training
GAMMA = 0.5  # Discount factor
C = 8  # C constant for the improved reward function
TARGET_UPDATE = 10
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_HIDDEN_LAYERS = 3
LEARNING_RATE = 1e-2
REWARD_PENALTY = 1.5
ENVIRONMENT_MEMORY = 2
MAX_NUMBER_OF_AGENTS = 5
ITERATIONS_PER_NUM_AGENTS = 50
NUM_ACTIONS = 10
max_d2d = MAX_NUMBER_OF_AGENTS
# more parameters
# linear discretization
# actions = power_to_db(np.linspace(
#     db_to_power(p_max-20), db_to_power(p_max-10), 10
# ))
# db discretization
actions = power_to_db(
    np.linspace(
        1e-6, db_to_power(p_max-10), NUM_ACTIONS
    )
)
env_params = EnvironmentParameters(
    rb_bandwidth, d2d_pair_distance, p_max, noise_power,
    bs_gain, user_gain, sinr_threshold_train,
    n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin
)
params = TrainingParameters(1, STEPS_PER_EPISODE)
agent_params = DQNAgentParameters(
    EPSILON_MIN, EPSILON_DECAY, EPSILON_INITIAL, REPLAY_MEMORY_SIZE,
    BATCH_SIZE, GAMMA
)
reward_function = dis_reward_tensor_db
channel_to_devices = BANChannel(rnd=CHANNEL_RND)
channel_to_bs = UrbanMacroNLOSWinnerChannel(
    rnd=CHANNEL_RND, f_c=carrier_frequency, h_bs=bs_height, h_ms=device_height
)
ref_env = CompleteEnvironment10dB(
    env_params,
    reward_function,
    channel_to_bs,
    channel_to_devices,
    reward_penalty=REWARD_PENALTY,
    memory=ENVIRONMENT_MEMORY,
    bs_height=bs_height
)
# foo env and foo agents stuff
foo_env = deepcopy(ref_env)
foo_agents = [ExternalDQNAgent(agent_params, [1]) for _ in range(4)]
foo_env.build_scenario(foo_agents)
_, _ = foo_env.step(foo_agents)
env_state_size = foo_env.get_state_size(foo_agents[0])


def calculate_interferences(env: CompleteEnvironment10dB):
    bs = env.bs
    mue = env.mue
    d2d_pairs = env.d2d_pairs
    txs = [mue]
    txs += [p[0] for p in d2d_pairs]
    rxs = [bs]
    rxs += [p[1] for p in d2d_pairs]
    interferences = np.zeros((len(txs), len(rxs)))
    for i, tx in enumerate(txs):
        for j, (rx, interfered) in enumerate(zip(rxs, txs)):
            if tx == interfered:
                interf = tx.power_at_receiver
            elif tx == mue:
                interf = interfered.received_mue_interference
            elif rx == bs:
                interf = tx.caused_mue_interference
            else:
                interf = [
                    i[1] for i in interfered.interferences
                    if i[0] == tx.id
                ][0]
            interferences[i][j] = interf
    tx_labels = [d.id for d in txs]
    rx_labels = [d.id for d in rxs]
    return interferences, tx_labels, rx_labels


def print_stuff(actions, env: CompleteEnvironment10dB):
    actions = [f'{i:.2f}' for i in actions]
    sinr_d2ds = [f'{d[0].sinr:.2f}' for d in env.d2d_pairs]
    print(f'MUE Tx Power [dBW]: {env.mue.tx_power:.2f}')
    print(f'D2D Power levels [dBW]: {actions}')
    print(f'D2D SINR [dB]: {sinr_d2ds}')
    print(f'D2D Spectral Efficiencies: {env.d2d_spectral_eff}')


def test(
    agents: List[ExternalDQNAgent],
):
    global actions
    n_agents = len(agents)
    actions_tuples = \
        list(product(range(len(actions)), repeat=n_agents))
    mue_spectral_effs = []
    d2d_spectral_effs = []
    rewards_bag = []
    rewards = []
    pathlosses_d2d_to_bs = []
    test_env = deepcopy(ref_env)
    test_env.build_scenario(agents)
    total_reward = 0.0
    best_reward = float('-inf')
    best_action_tuple = 0
    sample_size = \
        STEPS_PER_EPISODE \
        if STEPS_PER_EPISODE < len(actions_tuples) \
        else len(actions_tuples)
    sampled_action_tuples = \
        random.sample(actions_tuples, sample_size)
    # search for the best action_tuple
    # this should be in the `train` function
    # but i am too lazy for that, at the moment
    for i in sampled_action_tuples:
        action_tuple = i
        for j, agent in enumerate(agents):
            agent.set_action(action_tuple[j], actions[action_tuple[j]])
        _, rewards = test_env.step(agents)
        total_reward = sum(rewards)
        if total_reward > best_reward:
            best_reward = total_reward
            best_action_tuple = action_tuple
    # use the best action tuple throughout the whole testing
    for _ in range(TEST_STEPS_PER_EPISODE):
        action_tuple = best_action_tuple
        for j, agent in enumerate(agents):
            agent.set_action(action_tuple[j], actions[action_tuple[j]])
        _, rewards = test_env.step(agents)
        total_reward = sum(rewards)
        # saving stuff
        rewards_bag.append(total_reward)
        mue_spectral_effs.append(test_env.mue_spectral_eff.item())
        d2d_spectral_effs.append(test_env.d2d_spectral_eff.item())
        pathlosses_d2d_to_bs.append(
            test_env.total_losses[agents[0].id][test_env.bs.id]
        )
    mue_success_rate = np.mean(
        np.array(mue_spectral_effs) > np.log2(
            1 + db_to_power(sinr_threshold_train)
        )
    )
    # jain_index_avg = list()
    # for i, j in enumerate(jain_index):
    #     jain_index_avg.append(np.average(j))
    # save data
    return mue_success_rate, mue_spectral_effs, d2d_spectral_effs, rewards, \
        pathlosses_d2d_to_bs


def run():
    mue_sucess_rate_total = []
    mue_spectral_effs_total = []
    d2d_spectral_effs_total = []
    rewards_total = []
    pathlosses_d2d_to_bs = []
    start = time()
    for it in range(ITERATIONS_PER_NUM_AGENTS):
        now = (time() - start) / 60
        print(
            f'Number of agents: {1}. ' +
            f'Iteration: {it}/{ITERATIONS_PER_NUM_AGENTS-1}. ' +
            f'Elapsed time: {now} minutes.'
        )
        agents = [ExternalDQNAgent(agent_params, actions)
                  for _ in range(1)]
        mue_success_rate, mue_spectral_effs, d2d_spectral_effs, rewards, \
            pathlosses_d2d_to_bs = test(agents)
        mue_sucess_rate_total.append(mue_success_rate)
        mue_spectral_effs_total.append(mue_spectral_effs)
        d2d_spectral_effs_total.append(d2d_spectral_effs)
        rewards_total.append(rewards)
        pathlosses_d2d_to_bs.append(pathlosses_d2d_to_bs)
    # save stuff
    now = (time() - start) / 60
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    dir_path = f'data/benchmarks/{filename}'
    data_path = make_dir_timestamp(dir_path)
    data_file_path = f'{data_path}/log.pickle'
    data = {
        'mue_success_rate': mue_sucess_rate_total,
        'd2d_speffs': d2d_spectral_effs_total,
        'mue_speffs': mue_spectral_effs_total,
        'rewards': rewards_total,
        'mue_sinr_threshold': sinr_threshold_train,
        'elapsed_time': now,
        'pathlosses_d2d_to_bs': pathlosses_d2d_to_bs
    }
    save_with_pickle(data, data_file_path)
    copyfile(__file__, f'{data_path}/{filename}.py')
    print(f'done. Elapsed time: {now} minutes.')


if __name__ == '__main__':
    run()
