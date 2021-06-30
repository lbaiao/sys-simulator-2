# Similar to script .
# Uses CompleteEnvironment10dB
# Centralized Learning-Distributed Execution
# Simulates many times, for different number of agents, and take the averages.
# There are different channels to the BS and to the devices.
# Multiple episodes convergence. Everything is in dB.
# One NN is trained and copied to each agent.
from shutil import copyfile
from sys_simulator.general import make_dir_timestamp, save_with_pickle
import matplotlib.pyplot as plt
from sys_simulator.plots import plot_positions_actions_pie
from time import time
from sys_simulator.general import db_to_power, power_to_db
from sys_simulator.channels import BANChannel, UrbanMacroNLOSWinnerChannel
from sys_simulator import general as gen
from sys_simulator.q_learning.environments.completeEnvironment10dB \
    import CompleteEnvironment10dB
from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from sys_simulator.dqn.externalDQNFramework import ExternalDQNFramework
from sys_simulator.parameters.parameters import \
    EnvironmentParameters, TrainingParameters, DQNAgentParameters
from sys_simulator.q_learning.rewards import dis_reward_tensor_db
from copy import deepcopy
import torch
import numpy as np
import pickle


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
# exec params
STEPS_PER_EPISODE = 5
TEST_STEPS_PER_EPISODE = 5
MAX_NUM_EPISODES = 1500      # medium training
ITERATIONS_PER_NUM_AGENTS = 100
EVAL_EVERY = 80
EVAL_NUM_EPISODES = 100
EVAL_STEPS_PER_EPISODE = 5
# debug params
# STEPS_PER_EPISODE = 2
# TEST_STEPS_PER_EPISODE = 2
# MAX_NUM_EPISODES = 10
# ITERATIONS_PER_NUM_AGENTS = 10
# EVAL_EVERY = 1000
# EVAL_NUM_EPISODES = 2
# EVAL_STEPS_PER_EPISODE = 2
# common
EPSILON_INITIAL = 1
EPSILON_MIN = .05
# EPSILON_DECAY = .9*1e-4    # medium training
EPSILON_DECAY = 1.3 / (MAX_NUM_EPISODES * STEPS_PER_EPISODE)  # medium training
GAMMA = 0.9  # Discount factor
C = 1  # C constant for the improved reward function
TARGET_UPDATE = 10
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_HIDDEN_LAYERS = 1
LEARNING_RATE = 1e-2
REWARD_PENALTY = 1.5
ENVIRONMENT_MEMORY = 2
MAX_NUMBER_OF_AGENTS = 5
max_d2d = MAX_NUMBER_OF_AGENTS
range_n_d2d = range(1, max_d2d + 1, 1)
# more parameters
# linear discretization
# actions = power_to_db(np.linspace(
#     db_to_power(p_max-20), db_to_power(p_max-10), 10
# ))
# db discretization
actions = power_to_db(
    np.linspace(
        1e-6, db_to_power(p_max-10), 10
    )
)
env_params = EnvironmentParameters(
    rb_bandwidth, d2d_pair_distance, p_max, noise_power,
    bs_gain, user_gain, sinr_threshold_train,
    n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin
)
params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
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


def train(start):
    global actions
    framework = ExternalDQNFramework(
        agent_params,
        env_state_size,
        len(actions),
        HIDDEN_SIZE,
        NUM_HIDDEN_LAYERS,
        LEARNING_RATE
    )
    best_reward = float('-inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mue_spectral_eff_bag = list()
    d2d_spectral_eff_bag = list()
    rewards_bag = list()
    # aux_range = range(max_d2d+1)[1:]
    epsilon = agent_params.start_epsilon
    for episode in range(MAX_NUM_EPISODES):
        env = deepcopy(ref_env)
        n_agents = np.random.choice(range_n_d2d)
        now = (time() - start) / 60
        print(
            'Training. ' +
            f'Number of agents: {n_agents}. ' +
            f'Episode: {episode}/{MAX_NUM_EPISODES-1}. ' +
            f'Epsilon: {epsilon}. '
            f'Elapsed time: {now} minutes.'
        )
        agents = [ExternalDQNAgent(agent_params, actions)
                  for _ in range(n_agents)]  # 1 agent per d2d tx
        for a in agents:
            a.set_epsilon(epsilon)
        env.build_scenario(agents)
        obs, _ = env.step(agents)
        total_reward = 0.0
        i = 0
        bag = list()
        while True:
            if i >= params.steps_per_episode:
                break
            else:
                past_actions = torch.zeros([len(agents)], device=device)
                for j, agent in enumerate(agents):
                    agent.get_action(framework, obs[j].float())
                    past_actions[j] = agent.action_index
                # # debugging
                # if len(agents) == 2:
                #     print('debugging')
                # aux1 = agents[0].action_index == 9
                # aux2 = agents[1].action_index == 5
                # aux = [aux1, aux2]
                # if np.mean(aux) == 1:
                #     print('debugging')
                next_obs, rewards = env.step(agents)
                i += 1
                for j, agent in enumerate(agents):
                    framework.replay_memory.push(
                        obs[j].float(), past_actions[j],
                        next_obs[j].float(), rewards[j]
                    )
                framework.learn()
                total_reward = np.sum(rewards)
                bag.append(total_reward.item())
                obs = next_obs
                if i % TARGET_UPDATE == 0:
                    framework.target_net.load_state_dict(
                        framework.policy_net.state_dict()
                    )
                if total_reward > best_reward:
                    best_reward = total_reward
        epsilon = agents[0].epsilon
        if episode % EVAL_EVERY == 0:
            r, d_speff, m_speff = in_training_test(framework, device)
            rewards_bag.append(r)
            # average d2d spectral eff
            d2d_spectral_eff_bag.append(d_speff)
            # mue spectral eff
            mue_spectral_eff_bag.append(m_speff)
    # save stuff
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    data_path = f'models/dql/{filename}.pt'
    torch.save(framework.policy_net.state_dict(), data_path)
    # Return the trained policy
    return framework, rewards_bag, d2d_spectral_eff_bag, mue_spectral_eff_bag, epsilon  # noqa


def test(n_agents, test_env, framework):
    framework.policy_net.eval()
    mue_spectral_effs = []
    d2d_spectral_effs = []
    rewards_bag = []
    # jain_index = [list() for _ in range(max_d2d+1)]
    bag = list()
    agents = [ExternalDQNAgent(agent_params, actions)
              for i in range(n_agents)]  # 1 agent per d2d tx
    test_env.build_scenario(agents)
    obs, _ = test_env.step(agents)
    total_reward = 0.0
    i = 0
    while True:
        actions_index = list()
        for j, agent in enumerate(agents):
            aux = agent.act(framework, obs[j].float()).max(1)
            agent.set_action(aux[1].long(),
                             agent.actions[aux[1].item()])
            bag.append(aux[1].item())
            actions_index.append(aux[1].item())
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
    # jain_index_avg = list()
    # for i, j in enumerate(jain_index):
    #     jain_index_avg.append(np.average(j))
    # save data
    return mue_success_rate, mue_spectral_effs, d2d_spectral_effs, rewards


def in_training_test(framework: ExternalDQNFramework, device: torch.device):
    mue_spectral_eff_bag = list()
    d2d_spectral_eff_bag = list()
    rewards_bag = list()
    for _ in range(EVAL_NUM_EPISODES):
        env = deepcopy(ref_env)
        n_agents = np.random.choice(range_n_d2d)
        agents = [ExternalDQNAgent(agent_params, actions)
                  for _ in range(n_agents)]  # 1 agent per d2d tx
        env.build_scenario(agents)
        obs, _ = env.step(agents)
        for _ in range(EVAL_STEPS_PER_EPISODE):
            for j, agent in enumerate(agents):
                aux = agent.act(framework, obs[j].float()).max(1)
                agent.set_action(aux[1].long(),
                                 agent.actions[aux[1].item()])
            next_obs, _ = env.step(agents)
            obs = next_obs
            # mue spectral eff
            mue_spectral_eff_bag.append(env.mue_spectral_eff)
            # average d2d spectral eff
            d2d_spectral_eff_bag.append(env.d2d_spectral_eff)
            rewards_bag.append(env.reward)
    mean_mue_speff = np.mean(mue_spectral_eff_bag)
    mean_d2d_speff = np.mean(d2d_spectral_eff_bag)
    mean_reward = np.mean(rewards_bag)
    return mean_reward, mean_d2d_speff, mean_mue_speff


def run(framework=None):
    mue_sucess_rate_total = []
    mue_spectral_effs_total = []
    d2d_spectral_effs_total = []
    rewards_total = []
    start = time()
    r, d_speffs, m_speffs, epsilon = 0, 0, 0, 1
    if framework is None:
        framework, r, d_speffs, m_speffs, epsilon = train(start)
    for n in range(1, MAX_NUMBER_OF_AGENTS+1, 1):
        mue_suc_rates = []
        mue_speff_rates = []
        d2d_speff_rates = []
        rews = []
        for it in range(ITERATIONS_PER_NUM_AGENTS):
            now = (time() - start) / 60
            print(
                'Testing. ' +
                f'Number of agents: {n}/{MAX_NUMBER_OF_AGENTS}. ' +
                f'Iteration: {it}/{ITERATIONS_PER_NUM_AGENTS-1}. ' +
                f'Elapsed time: {now} minutes.'
            )
            test_env = deepcopy(ref_env)
            mue_success_rate, mue_spectral_effs, d2d_spectral_effs, rewards = \
                test(n, test_env, framework)
            mue_suc_rates.append(mue_success_rate)
            mue_speff_rates.append(mue_spectral_effs)
            d2d_speff_rates.append(d2d_spectral_effs)
            rews.append(rewards)
        mue_sucess_rate_total.append(mue_suc_rates)
        mue_spectral_effs_total.append(mue_speff_rates)
        d2d_spectral_effs_total.append(d2d_speff_rates)
        rewards_total.append(rews)
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
        'training_rewards': r,
        'training_d2d_speffs': d_speffs,
        'training_mue_speffs': m_speffs,
        'eval_every': EVAL_EVERY,
        'final_epsilon': epsilon,
    }
    save_with_pickle(data, data_file_path)
    copyfile(__file__, f'{data_path}/{filename}.py')
    print(f'done. Elapsed time: {now} minutes.')


def run_test():
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    data_path = f'models/dql/{filename}.pt'
    framework = torch.load(data_path)
    run(framework)


def test_exec():
    # environment
    test_env = deepcopy(ref_env)
    # load framework
    framework = ExternalDQNFramework(
        agent_params,
        env_state_size,
        len(actions),
        HIDDEN_SIZE,
        NUM_HIDDEN_LAYERS,
        LEARNING_RATE
    )
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    data_path = f'models/dql/{filename}.pt'
    state_dict = torch.load(data_path)
    framework.policy_net.load_state_dict(state_dict)
    framework.policy_net.eval()
    # simulation stuff
    mue_spectral_effs = []
    d2d_spectral_effs = []
    rewards_bag = []
    # devices positions
    pairs_positions = [
        ((-400, 0, device_height), (-450, 0, device_height)),
        ((100, 0, device_height), (150, 0, device_height)),
        ((225, 225, device_height), (275, 225, device_height)),
        ((55, -55, device_height), (55, -5, device_height)),
    ]
    mue_position = (0, 200, device_height)
    # jain_index = [list() for _ in range(max_d2d+1)]
    n_agents = len(pairs_positions)
    bag = list()
    agents = [ExternalDQNAgent(agent_params, actions)
              for i in range(n_agents)]  # 1 agent per d2d tx
    test_env.set_scenario(pairs_positions, mue_position, agents)
    obs, _ = test_env.step(agents)
    total_reward = 0.0
    i = 0
    while True:
        actions_index = list()
        for j, agent in enumerate(agents):
            aux = agent.act(framework, obs[j].float()).max(1)
            agent.set_action(aux[1].long(),
                             agent.actions[aux[1].item()])
            bag.append(aux[1].item())
            actions_index.append(aux[1].item())
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
    d2d_txs, d2d_rxs = zip(*test_env.d2d_pairs)
    # D2D interference on the MUE, in dB
    d2d_interferences = np.array([
        d.caused_mue_interference for d in d2d_txs
    ])
    d2d_interferences_mag = db_to_power(d2d_interferences)
    d2d_total_interference = np.sum(d2d_interferences_mag)
    percentage_interferences = d2d_interferences_mag / d2d_total_interference
    interferences, tx_labels, rx_labels = calculate_interferences(test_env)
    if d2d_total_interference != 0:
        plot_positions_actions_pie(
            test_env.bs, test_env.mue, d2d_txs, d2d_rxs,
            actions_index, percentage_interferences,
            test_env.mue.sinr > sinr_threshold_train, sinr_threshold_train,
            test_env.reward, interferences, tx_labels, rx_labels
        )
    # jain_index[n_agents].append(gen.jain_index(test_env.sinr_d2ds))
    mue_success_rate = np.mean(
        np.array(mue_spectral_effs) > np.log2(
            1 + db_to_power(sinr_threshold_train)
        )
    )
    # jain_index_avg = list()
    # for i, j in enumerate(jain_index):
    #     jain_index_avg.append(np.average(j))
    # save data
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    data_path = f'data/dql/{filename}_exec.pickle'
    data = {
        'd2d_speffs_avg_total': d2d_spectral_effs,
        'mue_success_rate': mue_success_rate,
        'chosen_actions': bag,
        'd2d_speffs': d2d_spectral_effs,
        'mue_speffs': mue_spectral_effs,
        'rewards': rewards_bag,
        'mue_sinr_threshold': sinr_threshold_train,
    }
    with open(data_path, 'wb') as file:
        pickle.dump(data, file)
    # plot
    print_stuff(actions, test_env)
    plt.show()


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


def print_stuff(actions, env: CompleteEnvironment10dB):
    actions = [f'{i:.2f}' for i in actions]
    sinr_d2ds = [f'{d[0].sinr:.2f}' for d in env.d2d_pairs]
    print(f'MUE Tx Power [dBW]: {env.mue.tx_power:.2f}')
    print(f'D2D Power levels [dBW]: {actions}')
    print(f'D2D SINR [dB]: {sinr_d2ds}')
    print(f'D2D Spectral Efficiencies: {env.d2d_spectral_eff}')


if __name__ == '__main__':
    run()
