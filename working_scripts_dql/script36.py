# Single episode convergence. Everything is in dB.
# Central DQN cointrols all agents.
from itertools import product
from sys_simulator.general import db_to_power, power_to_db
from sys_simulator.plots import plot_positions_actions_pie
from scipy.spatial.distance import euclidean
from sys_simulator.channels import BANChannel
from sys_simulator import general as gen
from sys_simulator.q_learning.environments.completeEnvironment7dB \
    import CompleteEnvironment7dB
from sys_simulator.dqn.agents.dqnAgent import CentralDQNAgent, ExternalDQNAgent
from sys_simulator.dqn.externalDQNFramework import ExternalDQNFramework
from sys_simulator.parameters.parameters import \
    EnvironmentParameters, TrainingParameters, DQNAgentParameters
from sys_simulator.q_learning.rewards import dis_reward_tensor_db
import torch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


n_mues = 1  # number of mues
n_d2d = 2  # number of d2d pairs
n_rb = n_mues   # number of RBs
bs_radius = 500  # bs radius in m
rb_bandwidth = 180*1e3  # rb bandwidth in Hz
d2d_pair_distance = 50  # d2d pair distance in m
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
CHANNEL_RND = False
# q-learning parameters
# training
NUMBER = 1
STEPS_PER_EPISODE = 1000
TEST_STEPS_PER_EPISODE = 200
# common
EPSILON_INITIAL = 1
EPSILON_MIN = .05
EPSILON_DECAY = 1.1 / STEPS_PER_EPISODE    # fast training
GAMMA = 0.5  # Discount factor
C = 8  # C constant for the improved reward function
TARGET_UPDATE = 10
MAX_NUMBER_OF_AGENTS = 4
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_HIDDEN_LAYERS = 2
LEARNING_RATE = 1e-2
max_d2d = MAX_NUMBER_OF_AGENTS
# more parameters
# linear discretization
# actions = power_to_db(np.linspace(
#     db_to_power(p_max-20), db_to_power(p_max-10), 10
# ))
# db discretization
NUM_ACTIONS = 10
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
channel = BANChannel(rnd=CHANNEL_RND)
env = CompleteEnvironment7dB(env_params, reward_function, channel, memory=2)
foo_agents = [ExternalDQNAgent(agent_params, [1]) for a in range(4)]
env.build_scenario(foo_agents)
_, _ = env.step(foo_agents)
env_state_size = env.get_state_size(foo_agents[0])
# pairs_positions = [
#     (450, 0),
#     (-350, 0),
#     (0, 150),
#     (0, -250)
# ]
pairs_positions = [
    ((-400, 0), (-450, 0)),
    ((100, 0), (150, 0))
]
# mue_position = (250 / math.sqrt(2), 250 / math.sqrt(2))
mue_position = (0, 200)
n_agents = len(pairs_positions)
actions_tuples = \
    list(product(range(NUM_ACTIONS), repeat=n_agents))
framework = ExternalDQNFramework(
    agent_params,
    env_state_size * n_agents,
    len(actions_tuples),
    HIDDEN_SIZE,
    NUM_HIDDEN_LAYERS,
    LEARNING_RATE
)


def calculate_interferences(env: CompleteEnvironment7dB):
    bs = env.bs
    mue = env.mue
    d2d_pairs = env.d2d_pairs
    txs = [mue]
    txs += [p[0] for p in d2d_pairs]
    rxs = [bs]
    rxs += [p[1] for p in d2d_pairs]
    interferences = np.zeros((len(txs), len(rxs)))
    for i, tx in enumerate(txs):
        for j, rx in enumerate(rxs):
            interf = \
                tx.tx_power + tx.gain \
                - env.channel.step(euclidean(tx.position, rx.position)) \
                + rx.gain
            interferences[i][j] = interf
    tx_labels = [d.id for d in txs]
    rx_labels = [d.id for d in rxs]
    return interferences, tx_labels, rx_labels


def train():
    global actions
    best_reward = float('-inf')
    mue_spectral_eff_bag = list()
    d2d_spectral_eff_bag = list()
    # aux_range = range(max_d2d+1)[1:]
    epsilon = agent_params.start_epsilon
    # n_agents = np.random.choice(aux_range)
    agents = [ExternalDQNAgent(agent_params, actions)
              for _ in range(n_agents)]  # 1 agent per d2d tx
    central_agent = CentralDQNAgent(agent_params, actions, n_agents)
    central_agent.set_epsilon(epsilon)
    env.set_scenario(pairs_positions, mue_position, agents)
    obs_aux, _ = env.step(agents)
    obs = torch.cat(obs_aux).view(1, -1).float()
    # env.build_scenario(agents)
    # obs = [env.get_state(a).float() for a in agents]
    reward = 0.0
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
            # # debugging
            # if len(agents) == 2:
            #     print('debugging')
            next_obs_aux, rewards = env.step(agents)
            next_obs = torch.cat(next_obs_aux).view(1, -1).float()
            reward = np.sum(rewards)
            i += 1
            framework.replay_memory.push(
                obs, tuple_index, next_obs, reward
            )
            framework.learn()
            bag.append(reward)
            obs = next_obs
            if i % TARGET_UPDATE == 0:
                framework.target_net.load_state_dict(
                    framework.policy_net.state_dict()
                )
            if reward > best_reward:
                best_reward = reward
            print("Step#:{} sum reward:{} best_sum_reward:{} eps:{}".format(
                i, reward, best_reward, central_agent.epsilon)
            )
        # mue spectral eff
        mue_spectral_eff_bag.append(
            (env.mue_spectral_eff, n_agents)
        )
        # average d2d spectral eff
        d2d_spectral_eff_bag.append(
            (env.d2d_spectral_eff/n_agents, n_agents)
        )
    epsilon = central_agent.epsilon
    # Return the trained policy
    mue_spectral_effs = mue_spectral_eff_bag
    d2d_spectral_effs = d2d_spectral_eff_bag
    spectral_effs = zip(mue_spectral_effs, d2d_spectral_effs)
    avg_q_values = framework.bag
    # # saving the data and the model
    cwd = os.getcwd()
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    filename_model = filename
    filename = f'{cwd}/data/dql/{filename}_training.pt'
    torch.save(framework.policy_net.state_dict(),
               f'{cwd}/models/dql/{filename_model}.pt')
    torch.save(spectral_effs, filename)
    with open(
        f'{cwd}/data/dql/{filename_model}_avg_q_values.pickle',
        'wb'
    ) as p_file:
        pickle.dump(avg_q_values, p_file)


def print_stuff(actions, env: CompleteEnvironment7dB):
    actions = [f'{i:.2f}' for i in actions]
    sinr_d2ds = [f'{d[0].sinr:.2f}' for d in env.d2d_pairs]
    print(f'MUE Tx Power [dBW]: {env.mue.tx_power:.2f}')
    print(f'D2D Power levels [dBW]: {actions}')
    print(f'D2D SINR [dB]: {sinr_d2ds}')
    print(f'D2D Spectral Efficiencies: {env.d2d_spectral_eff}')


def test():
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    framework.policy_net.load_state_dict(
        torch.load(f'models/dql/{filename}.pt')
    )
    framework.policy_net.eval()
    mue_spectral_effs = [list() for _ in range(max_d2d+1)]
    d2d_spectral_effs = [list() for _ in range(max_d2d+1)]
    # jain_index = [list() for _ in range(max_d2d+1)]
    # done = False
    bag = list()
    # aux_range = range(max_d2d+1)[1:]
    # n_agents = np.random.choice(aux_range)
    agents = [ExternalDQNAgent(agent_params, actions)
              for i in range(n_agents)]  # 1 agent per d2d tx
    central_agent = CentralDQNAgent(agent_params, actions, n_agents)
    env.set_scenario(pairs_positions, mue_position, agents)
    # env.build_scenario(agents)
    # done = False
    # obs = [env.get_state(a) for a in agents]
    obs_aux, _ = env.step(agents)
    obs = torch.cat(obs_aux).view(1, -1).float()
    total_reward = 0.0
    i = 0
    while True:
        tuple_index = central_agent.act(framework, obs)
        action_tuple = actions_tuples[tuple_index]
        actions_index = actions_tuples[tuple_index]
        bag.append(tuple_index)
        for j, agent in enumerate(agents):
            agent.set_action(action_tuple[j], actions[action_tuple[j]])
        next_obs_aux, rewards = env.step(agents)
        next_obs = torch.cat(next_obs_aux).view(1, -1).float()
        obs = next_obs
        total_reward += sum(rewards)
        # saving stuff
        mue_spectral_effs[n_agents].append(env.mue_spectral_eff.item())
        d2d_spectral_effs[n_agents].append(env.d2d_spectral_eff.item())
        i += 1
        if i >= TEST_STEPS_PER_EPISODE:
            break
    d2d_txs, d2d_rxs = zip(*env.d2d_pairs)
    # D2D interference on the MUE, in dB
    d2d_interferences = np.array([
        d.tx_power + env.params.user_gain + env.params.bs_gain -
        env.channel.step(d.distance_to_bs) for d in d2d_txs
    ])
    d2d_interferences_mag = db_to_power(d2d_interferences)
    d2d_total_interference = np.sum(d2d_interferences_mag)
    percentage_interferences = d2d_interferences_mag / d2d_total_interference
    interferences, tx_labels, rx_labels = calculate_interferences(env)
    if d2d_total_interference != 0:
        plot_positions_actions_pie(
            env.bs, env.mue, d2d_txs, d2d_rxs,
            actions_index, percentage_interferences,
            env.mue.sinr > sinr_threshold_train, sinr_threshold_train,
            env.reward, interferences, tx_labels, rx_labels
        )
    print_stuff(actions, env)
    plt.show()
    # jain_index[n_agents].append(gen.jain_index(env.sinr_d2ds))
    mue_success_rate = list()
    for i, m in enumerate(mue_spectral_effs):
        mue_success_rate.append(
            np.average(m > np.log2(1 + sinr_threshold_train))
        )
    d2d_speffs_avg = list()
    for i, d in enumerate(d2d_spectral_effs):
        d2d_speffs_avg.append(np.average(d))
    # jain_index_avg = list()
    # for i, j in enumerate(jain_index):
    #     jain_index_avg.append(np.average(j))
    log = list()
    for i, d in enumerate(zip(d2d_speffs_avg, mue_success_rate)):
        log.append(f'NUMBER OF D2D_USERS: {i+1}')
        log.append(f'D2D SPECTRAL EFFICIENCY - SCRIPT: {d[0]}')
        log.append(f'MUE SUCCESS RATE - SCRIPT: {d[1]}')
        log.append('-------------------------------------')
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    log_path = f'logs/dql/{filename}.txt'
    file = open(log_path, 'w')
    for lg in log:
        file.write(f'{lg}\n')
    file.close()
    data_path = f'data/dql/{filename}.pickle'
    data = {
        'd2d_speffs_avg_total': d2d_spectral_effs,
        'mue_success_rate': mue_success_rate,
        'chosen_actions': bag,
    }
    with open(data_path, 'wb') as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    train()
    test()
