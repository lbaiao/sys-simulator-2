# Uses CompleteEnvironment10dB
# Centralized Learning-Centralized Execution
# Simulate one episode. Take the rewards, actinos and pathloss to prove
# how dynamic the scenario is.
# There are different channels to the BS and to the devices.
# Single episode convergence. Everything is in dB.
# Central DQN controls all agents.
from shutil import copyfile
from itertools import product
from time import time
from typing import List, Tuple
from sys_simulator.general \
    import db_to_power, make_dir_timestamp, power_to_db, random_seed, save_with_pickle
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


RND_SEED = 42
random_seed(RND_SEED)
np.random.seed
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
MAX_NUMBER_OF_AGENTS = 1
ITERATIONS_PER_NUM_AGENTS = 1
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


def train(n_agents, env):
    global actions
    actions_tuples = \
        list(product(range(len(actions)), repeat=n_agents))
    framework = ExternalDQNFramework(
        agent_params,
        env_state_size * n_agents,
        len(actions_tuples),
        HIDDEN_SIZE,
        NUM_HIDDEN_LAYERS,
        LEARNING_RATE
    )
    best_reward = float('-inf')
    mue_spectral_eff_bag = list()
    d2d_spectral_eff_bag = list()
    rewards_bag = list()
    # aux_range = range(max_d2d+1)[1:]
    epsilon = agent_params.start_epsilon
    # n_agents = np.random.choice(aux_range)
    agents = [ExternalDQNAgent(agent_params, actions)
              for _ in range(n_agents)]  # 1 agent per d2d tx
    central_agent = CentralDQNAgent(agent_params, actions, n_agents)
    central_agent.set_epsilon(epsilon)
    for a in agents:
        a.set_epsilon(epsilon)
    env.build_scenario(agents)
    obs_aux, _ = env.step(agents)
    obs = torch.cat(obs_aux).view(1, -1).float()
    # env.build_scenario(agents)
    # obs = [env.get_state(a).float() for a in agents]
    total_reward = 0.0
    i = 0
    bag = list()
    while True:
        if i >= params.steps_per_episode:
            break
        else:
            tuple_index = central_agent.get_action(framework, obs).item()
            action_tuple = actions_tuples[tuple_index]
            for j, agent in enumerate(agents):
                agent.set_action(action_tuple[j], actions[action_tuple[j]])
            next_obs_aux, rewards = env.step(agents)
            total_reward = np.sum(rewards)
            next_obs = torch.cat(next_obs_aux).view(1, -1).float()
            i += 1
            framework.replay_memory.push(
                obs, tuple_index, next_obs, total_reward
            )
            framework.learn()
            bag.append(total_reward.item())
            obs = next_obs
            if i % TARGET_UPDATE == 0:
                framework.target_net.load_state_dict(
                    framework.policy_net.state_dict()
                )
            if total_reward > best_reward:
                best_reward = total_reward
            # mue spectral eff
            mue_spectral_eff_bag.append(env.mue_spectral_eff)
            # average d2d spectral eff
            d2d_spectral_eff_bag.append(env.d2d_spectral_eff)
            rewards_bag.append(env.reward)
            # print("Step#:{} sum reward:{} best_sum_reward:{} eps:{}".format(
            #     i, total_reward, best_reward, agents[0].epsilon)
            # )
    epsilon = central_agent.epsilon
    # Return the trained policy
    return framework, central_agent, agents, actions_tuples


def print_stuff(actions, env: CompleteEnvironment10dB):
    actions = [f'{i:.2f}' for i in actions]
    sinr_d2ds = [f'{d[0].sinr:.2f}' for d in env.d2d_pairs]
    print(f'MUE Tx Power [dBW]: {env.mue.tx_power:.2f}')
    print(f'D2D Power levels [dBW]: {actions}')
    print(f'D2D SINR [dB]: {sinr_d2ds}')
    print(f'D2D Spectral Efficiencies: {env.d2d_spectral_eff}')


def test(
    test_env: CompleteEnvironment10dB,
    framework: ExternalDQNFramework,
    central_agent: CentralDQNAgent,
    agents: List[ExternalDQNAgent],
    actions_tuples: Tuple
):
    framework.policy_net.eval()
    mue_spectral_effs = []
    d2d_spectral_effs = []
    rewards_bag = []
    actions_bag = []
    pathlosses_d2d_to_bs = []
    test_env.reset_before_build_set()
    obs_aux, _ = test_env.step(agents)
    pathlosses_d2d_to_bs.append(
        test_env.total_losses[agents[0].id][test_env.bs.id]
    )
    obs = torch.cat(obs_aux).view(1, -1).float()
    total_reward = 0.0
    i = 0
    while True:
        tuple_index = central_agent.act(framework, obs)
        action_tuple = actions_tuples[tuple_index]
        for j, agent in enumerate(agents):
            agent.set_action(action_tuple[j], actions[action_tuple[j]])
        next_obs_aux, rewards = test_env.step(agents)
        next_obs = torch.cat(next_obs_aux).view(1, -1).float()
        obs = next_obs
        total_reward = sum(rewards)
        # saving stuff
        rewards_bag.append(total_reward)
        mue_spectral_effs.append(test_env.mue_spectral_eff.item())
        d2d_spectral_effs.append(test_env.d2d_spectral_eff.item())
        actions_bag.append(action_tuple)
        pathlosses_d2d_to_bs.append(
            test_env.total_losses[agents[0].id][test_env.bs.id]
        )
        i += 1
        if i >= TEST_STEPS_PER_EPISODE:
            break
    mue_success_rate = np.mean(
        np.array(mue_spectral_effs) > np.log2(
            1 + db_to_power(sinr_threshold_train)
        )
    )
    # jain_index_avg = list()
    # for i, j in enumerate(jain_index):
    #     jain_index_avg.append(np.average(j))
    # save data
    return mue_success_rate, mue_spectral_effs, d2d_spectral_effs, \
        rewards_bag, actions_bag, pathlosses_d2d_to_bs[:-1]


def run():
    mue_sucess_rate_total = []
    mue_spectral_effs_total = []
    d2d_spectral_effs_total = []
    rewards_total = []
    pathlosses_d2d_to_bs_total = []
    actions_total = []
    start = time()
    for n in range(1, MAX_NUMBER_OF_AGENTS+1, 1):
        mue_suc_rates = []
        mue_speff_rates = []
        d2d_speff_rates = []
        rews = []
        path_d2d_bs = []
        for it in range(ITERATIONS_PER_NUM_AGENTS):
            now = (time() - start) / 60
            print(
                f'Number of agents: {n}/{MAX_NUMBER_OF_AGENTS}. ' +
                f'Iteration: {it}/{ITERATIONS_PER_NUM_AGENTS-1}. ' +
                f'Elapsed time: {now} minutes.'
            )
            env = deepcopy(ref_env)
            framework, central_agent, agents, actions_tuples = \
                train(n, env)
            mue_success_rate, mue_spectral_effs, d2d_spectral_effs, rewards, \
                actions_bag, pathlosses_d2d_to_bs = \
                test(env, framework,
                     central_agent, agents, actions_tuples)
            mue_suc_rates.append(mue_success_rate)
            mue_speff_rates.append(mue_spectral_effs)
            d2d_speff_rates.append(d2d_spectral_effs)
            rews.append(rewards)
            actions_total.append(actions_bag)
            path_d2d_bs.append(pathlosses_d2d_to_bs)
        mue_sucess_rate_total.append(mue_suc_rates)
        mue_spectral_effs_total.append(mue_speff_rates)
        d2d_spectral_effs_total.append(d2d_speff_rates)
        rewards_total.append(rews)
        pathlosses_d2d_to_bs_total.append(path_d2d_bs)
    # save stuff
    now = (time() - start) / 60
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    dir_path = f'data/dql/{filename}'
    data_path = make_dir_timestamp(dir_path)
    data_file_path = f'{data_path}/log.pickle'
    data = {
        'mue_success_rate': mue_sucess_rate_total,
        'd2d_speffs': d2d_spectral_effs_total,
        'mue_speffs': mue_spectral_effs_total,
        'rewards': rewards_total,
        'mue_sinr_threshold': sinr_threshold_train,
        'elapsed_time': now,
        'pathlosses_d2d_to_bs': pathlosses_d2d_to_bs_total,
        'actions': actions_total
    }
    save_with_pickle(data, data_file_path)
    copyfile(__file__, f'{data_path}/{filename}.py')
    print(f'done. Elapsed time: {now} minutes.')


if __name__ == '__main__':
    run()
