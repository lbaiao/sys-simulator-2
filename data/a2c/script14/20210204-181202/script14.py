# Similar to scripts_dql/script39.py, but with A2C.
# Continuous-value actions.
# CompleteEnvironment10dB
# Simulates many times, for different number of agents, and take the averages.
# There are different channels to the BS and to the devices.
# Distributed learning-Distributed execution
# Single episode convergence. Everything is in dB. One NN for each agent.
from shutil import copyfile
import os
from sys_simulator.a2c.framework import ContinuousFramework
from sys_simulator.plots import plot_positions_actions_pie
from sys_simulator.a2c.agent import Agent
from sys_simulator.a2c import ActorCriticDiscrete
from typing import List
from sys_simulator.general import db_to_power, make_dir_timestamp, power_to_db, save_with_pickle
from sys_simulator.channels import BANChannel, UrbanMacroLOSWinnerChannel
from sys_simulator import general as gen
from sys_simulator.q_learning.environments.completeEnvironment10dB \
    import CompleteEnvironment10dB
from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from sys_simulator.parameters.parameters import \
    EnvironmentParameters, TrainingParameters, DQNAgentParameters
from sys_simulator.q_learning.rewards import dis_reward_tensor_db
from copy import deepcopy
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt


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
p_min = -60
p_max = p_max - 30
d2d_p_max = p_max - 10
noise_power = noise_power - 30
# channel parameters
CHANNEL_RND = True
# q-learning parameters
# training
NUMBER = 1
# run params
STEPS_PER_EPISODE = 20
TEST_STEPS_PER_EPISODE = 100
MAX_NUM_EPISODES = 70
# debugging params
# STEPS_PER_EPISODE = 2
# TEST_STEPS_PER_EPISODE = 2
# MAX_NUM_EPISODES = 10
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
NUM_HIDDEN_LAYERS = 1
LEARNING_RATE = 1e-5
BETA = 1e-2
REWARD_PENALTY = 1.5
ENVIRONMENT_MEMORY = 2
MAX_NUMBER_OF_AGENTS = 5
ITERATIONS_PER_NUM_AGENTS = 30
# ITERATIONS_PER_NUM_AGENTS = 10
max_d2d = MAX_NUMBER_OF_AGENTS
# more parameters
# linear discretization
# actions = power_to_db(np.linspace(
#     db_to_power(p_max-20), db_to_power(p_max-10), 10
# ))
# db discretization
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
channel_to_bs = UrbanMacroLOSWinnerChannel(
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
pairs_positions = [
    ((-400, 0, device_height), (-450, 0, device_height)),
    ((100, 0, device_height), (150, 0, device_height)),
    ((225, 225, device_height), (275, 225, device_height)),
    ((55, -55, device_height), (55, -5, device_height)),
]
mue_position = (0, 200, device_height)
n_agents = len(pairs_positions)


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
                    power_to_db(i[1]) for i in interfered.interferences
                    if i[0] == tx.id
                ][0]
            interferences[i][j] = interf
    tx_labels = [d.id for d in txs]
    rx_labels = [d.id for d in rxs]
    return interferences, tx_labels, rx_labels


def train(env: CompleteEnvironment10dB):
    mue_spectral_eff_bag = list()
    d2d_spectral_eff_bag = list()
    rewards_bag = list()
    best_reward = float('-inf')
    frameworks = [
        ContinuousFramework(
            env_state_size,
            1,
            HIDDEN_SIZE,
            NUM_HIDDEN_LAYERS,
            1e-9,
            1,
            STEPS_PER_EPISODE,
            LEARNING_RATE,
            BETA,
            GAMMA
        )
        for _ in range(n_agents)
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agents = [Agent(p_max) for _ in range(n_agents)]
    env.set_scenario(pairs_positions, mue_position, agents)
    for episode in range(MAX_NUM_EPISODES):
        obs, _ = env.step(agents)
        for i in range(STEPS_PER_EPISODE):
            for j, (agent, f) in enumerate(zip(agents, frameworks)):
                action, dist, value, mu, var = \
                    agent.act_continuous(f, obs[j])
                f.actions[0][i] = action
                f.mu[0][i] = mu
                f.vars[0][i] = var
                # log_prob = dist.log_prob(action)
                # f.log_probs[0][i] = log_prob
                f.values[0][i] = value
            # perform an environment step5
            next_obs_t, rewards_t = env.step(agents)
            for r, f in zip(rewards_t, frameworks):
                f.rewards[0][i] = r
            total_reward = np.sum(rewards_t)
            best_reward = \
                total_reward if total_reward > best_reward else best_reward
            obs = next_obs_t
            # mue spectral eff
            mue_spectral_eff_bag.append(env.mue_spectral_eff)
            # average d2d spectral eff
            d2d_spectral_eff_bag.append(env.d2d_spectral_eff)
            rewards_bag.append(env.reward)
            print(
                "Episode#:{}. Step#:{}. sum reward:{}. best sumreward:{}."
                .format(
                    episode, i, total_reward,
                    best_reward
                )
            )
        # gae and returns
        next_obs_t = torch.cat(obs, 0).to(device)
        for j, (agent, f) in enumerate(zip(agents, frameworks)):
            _, _, next_value_t, _, _ = agent.act_continuous(f, next_obs_t[j])
            f.values[0][STEPS_PER_EPISODE] = next_value_t
            f.learn()
    # Return the trained policy
    mue_spectral_effs = mue_spectral_eff_bag
    d2d_spectral_effs = d2d_spectral_eff_bag
    spectral_effs = zip(mue_spectral_effs, d2d_spectral_effs)
    # # saving the data and the model
    cwd = os.getcwd()
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    filename_model = filename
    filename = f'{cwd}/data/a2c/{filename}_training.pt'
    torch.save(frameworks[0].a2c.state_dict(),
               f'{cwd}/models/a2c/{filename_model}.pt')
    torch.save(spectral_effs, filename)
    with open(
        f'{cwd}/data/a2c/{filename_model}_rewards.pickle',
        'wb'
    ) as p_file:
        pickle.dump(rewards_bag, p_file)
    return frameworks, agents


def print_stuff(actions, env: CompleteEnvironment10dB):
    actions = [f'{i:.2f}' for i in actions]
    sinr_d2ds = [f'{d[0].sinr:.2f}' for d in env.d2d_pairs]
    print(f'MUE Tx Power [dBW]: {env.mue.tx_power:.2f}')
    print(f'D2D Power levels [dBW]: {actions}')
    print(f'D2D SINR [dB]: {sinr_d2ds}')
    print(f'D2D Spectral Efficiencies: {env.d2d_spectral_eff}')


def test(
    test_env: CompleteEnvironment10dB,
    frameworks: List[ActorCriticDiscrete],
    agents: List[Agent]
):
    for f in frameworks:
        f.a2c.eval()
    mue_spectral_effs = []
    d2d_spectral_effs = []
    rewards_bag = []
    # jain_index = [list() for _ in range(max_d2d+1)]
    bag = list()  # 1 agent per d2d tx
    test_env.reset_before_build_set()
    obs, _ = test_env.step(agents)
    total_reward = 0.0
    i = 0
    while True:
        actions_index = list()
        for j, (agent, framework) in enumerate(zip(agents, frameworks)):
            action, _, _, _, _ = agent.act_continuous(framework, obs[j])
            agent.set_action(action)
            bag.append(action)
            actions_index.append(action)
        next_obs, rewards = test_env.step(agents)
        obs = next_obs
        total_reward = sum(rewards)
        # saving stuff
        rewards_bag.append(total_reward)
        mue_spectral_effs.append(test_env.mue_spectral_eff.item())
        d2d_spectral_effs.append(test_env.d2d_spectral_eff.item())
        i += 1
        if i >= TEST_STEPS_PER_EPISODE:
            break
    mue_success_rate = np.mean(
        np.array(mue_spectral_effs) > np.log2(
            1 + db_to_power(sinr_threshold_train)
        )
    )
    d2d_txs, d2d_rxs = zip(*test_env.d2d_pairs)
    # D2D interference on the MUE, in dB
    d2d_interferences = np.array([
        d.caused_mue_interference for d in d2d_txs
    ])
    d2d_interferences_mag = db_to_power(d2d_interferences)
    d2d_total_interference = np.sum(d2d_interferences_mag)
    percentage_interferences = d2d_interferences_mag / d2d_total_interference
    interferences, tx_labels, rx_labels = calculate_interferences(test_env)
    actions_index = [f'{a:.2f}' for a in actions_index]
    if d2d_total_interference != 0:
        plot_positions_actions_pie(
            test_env.bs, test_env.mue, d2d_txs, d2d_rxs,
            actions_index, percentage_interferences,
            test_env.mue.sinr > sinr_threshold_train, sinr_threshold_train,
            test_env.reward, interferences, tx_labels, rx_labels
        )
    # jain_index[n_agents].append(gen.jain_index(test_env.sinr_d2ds))
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    dir_path = f'data/a2c/{filename}'
    data_path = make_dir_timestamp(dir_path)
    data_file_path = f'{data_path}/log.pickle'
    data = {
        'd2d_speffs_avg_total': d2d_spectral_effs,
        'mue_success_rate': mue_success_rate,
        'chosen_actions': bag,
        'd2d_speffs': d2d_spectral_effs,
        'mue_speffs': mue_spectral_effs,
        'rewards': rewards_bag,
        'mue_sinr_threshold': sinr_threshold_train
    }
    save_with_pickle(data, data_file_path)
    copyfile(__file__, f'{data_path}/{filename}.py')
    # print(f'done. Elapsed time: {now} minutes.')
    plt.show()
    # debugging
    # if mue_success_rate < 1:
    #     print('bug')
    # jain_index_avg = list()
    # for i, j in enumerate(jain_index):
    #     jain_index_avg.append(np.average(j))
    # save data


def run():
    env = deepcopy(ref_env)
    frameworks, agents = train(env)
    test(env, frameworks, agents)


if __name__ == '__main__':
    run()
