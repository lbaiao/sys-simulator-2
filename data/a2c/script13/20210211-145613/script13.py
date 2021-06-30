# is passed from training to testing, after being reset.
# Uses CompleteEnvironment10dB.
# Centralized Learning-Distributed Execution
# There are different channels to the BS and to the devices.
# Single episode convergence. Everything is in dB.
from shutil import copyfile
from sys_simulator.a2c.framework import DiscreteFramework
from sys_simulator.a2c.agent import Agent
from time import time
from sys_simulator.general import (
    db_to_power, make_dir_timestamp,
    power_to_db, save_with_pickle
)
from sys_simulator.channels import BANChannel, UrbanMacroNLOSWinnerChannel
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
REWARD_FUNCTION = 'classic'
# channel parameters
CHANNEL_RND = True
# q-learning parameters
# training
NUMBER = 1
# run params
STEPS_PER_EPISODE = 25
TEST_STEPS_PER_EPISODE = 25
MAX_NUM_EPISODES = 200
EVAL_EVERY = 20
EVAL_NUM_EPISODES = 100
EVAL_STEPS_PER_EPISODE = 25
ITERATIONS_PER_NUM_AGENTS = 50
# debugging params
# STEPS_PER_EPISODE = 20
# TEST_STEPS_PER_EPISODE = 20
# MAX_NUM_EPISODES = 20
# EVAL_EVERY = 10
# EVAL_NUM_EPISODES = 2
# EVAL_STEPS_PER_EPISODE = 20
# ITERATIONS_PER_NUM_AGENTS = 2
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
LEARNING_RATE = 1e-2
BETA = 1e-2
REWARD_PENALTY = 1.5
ENVIRONMENT_MEMORY = 2
MAX_NUMBER_OF_AGENTS = 5
# ITERATIONS_PER_NUM_AGENTS = 10
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
    channel_to_bs,
    channel_to_devices,
    reward_penalty=REWARD_PENALTY,
    memory=ENVIRONMENT_MEMORY,
    bs_height=bs_height,
    reward_function=REWARD_FUNCTION
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


def train(start):
    global actions
    framework = DiscreteFramework(
        env_state_size,
        len(actions),
        HIDDEN_SIZE,
        NUM_HIDDEN_LAYERS,
        1,
        STEPS_PER_EPISODE,
        LEARNING_RATE,
        BETA,
        GAMMA
    )
    best_reward = float('-inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mue_spectral_eff_bag = list()
    d2d_spectral_eff_bag = list()
    rewards_bag = list()
    for episode in range(MAX_NUM_EPISODES):
        env = deepcopy(ref_env)
        n_agents = np.random.choice(range_n_d2d)
        framework.reset_values(n_agents)
        now = (time() - start) / 60
        print(
            'Training. ' +
            f'Number of agents: {n_agents}. ' +
            f'Episode: {episode}/{MAX_NUM_EPISODES-1}. ' +
            f'Elapsed time: {now} minutes.'
        )
        agents = [Agent() for _ in range(n_agents)]
        env.build_scenario(agents)
        obs, _ = env.step(agents)
        total_reward = 0.0
        for i in range(STEPS_PER_EPISODE):
            for j, agent in enumerate(agents):
                action_index, dist, value = \
                    agent.act_discrete(framework, obs[j])
                agent.set_action(actions[action_index.item()])
                log_prob = dist.log_prob(action_index)
                framework.log_probs[j][i] = log_prob
                framework.values[j][i] = value
                framework.entropy[j][i] = dist.entropy()
            # perform an environment step
            next_obs_t, rewards_t = env.step(agents)
            for j, r in enumerate(rewards_t):
                framework.rewards[j][i] = r
            total_reward = np.sum(rewards_t)
            best_reward = \
                total_reward if total_reward > best_reward else best_reward
            obs = next_obs_t
        # gae and returns
        next_obs_t = torch.cat(obs, 0).to(device)
        for j, agent in enumerate(agents):
            _, _, next_value_t = agent.act_discrete(framework, next_obs_t[j])
            framework.values[0][STEPS_PER_EPISODE] = next_value_t
            framework.learn()
        if episode % EVAL_EVERY == 0:
            r, d_speff, m_speff = in_training_test(framework)
            rewards_bag.append(r)
            # average d2d spectral eff
            d2d_spectral_eff_bag.append(d_speff)
            # mue spectral eff
            mue_spectral_eff_bag.append(m_speff)
    # save stuff
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    data_path = f'models/a2c/{filename}.pt'
    torch.save(framework, data_path)
    # Return the trained policy
    return framework, rewards_bag, d2d_spectral_eff_bag, mue_spectral_eff_bag  # noqa


def in_training_test(framework: DiscreteFramework):
    mue_spectral_eff_bag = list()
    d2d_spectral_eff_bag = list()
    rewards_bag = list()
    for _ in range(EVAL_NUM_EPISODES):
        env = deepcopy(ref_env)
        n_agents = np.random.choice(range_n_d2d)
        agents = [Agent() for _ in range(n_agents)]
        env.build_scenario(agents)
        obs, _ = env.step(agents)
        for _ in range(EVAL_STEPS_PER_EPISODE):
            for j, agent in enumerate(agents):
                action_index, _, _ = agent.act_discrete(framework, obs[j])
                agent.set_action(actions[action_index.item()])
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


def print_stuff(actions, env: CompleteEnvironment10dB):
    actions = [f'{i:.2f}' for i in actions]
    sinr_d2ds = [f'{d[0].sinr:.2f}' for d in env.d2d_pairs]
    print(f'MUE Tx Power [dBW]: {env.mue.tx_power:.2f}')
    print(f'D2D Power levels [dBW]: {actions}')
    print(f'D2D SINR [dB]: {sinr_d2ds}')
    print(f'D2D Spectral Efficiencies: {env.d2d_spectral_eff}')


def test(
    n_agents: int,
    test_env: CompleteEnvironment10dB,
    framework: DiscreteFramework,
):
    framework.a2c.eval()
    mue_spectral_effs = []
    d2d_spectral_effs = []
    rewards_bag = []
    # jain_index = [list() for _ in range(max_d2d+1)]
    bag = list()
    agents = [Agent() for _ in range(n_agents)]
    test_env.build_scenario(agents)
    obs, _ = test_env.step(agents)
    total_reward = 0.0
    i = 0
    while True:
        for j, agent in enumerate(agents):
            action_index, _, _ = agent.act_discrete(framework, obs[j])
            agent.set_action(actions[action_index.item()])
            bag.append(action_index.item())
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
    return mue_success_rate, mue_spectral_effs, d2d_spectral_effs, rewards


def run(framework=None):
    mue_sucess_rate_total = []
    mue_spectral_effs_total = []
    d2d_spectral_effs_total = []
    rewards_total = []
    start = time()
    r, d_speffs, m_speffs = 0, 0, 0
    if framework is None:
        framework, r, d_speffs, m_speffs = train(start)
    for n in range(1, MAX_NUMBER_OF_AGENTS+1, 1):
        mue_suc_rates = []
        mue_speff_rates = []
        d2d_speff_rates = []
        rews = []
        for it in range(ITERATIONS_PER_NUM_AGENTS):
            now = (time() - start) / 60
            print(
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
    dir_path = f'data/a2c/{filename}'
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
    }
    save_with_pickle(data, data_file_path)
    copyfile(__file__, f'{data_path}/{filename}.py')
    print(f'done. Elapsed time: {now} minutes.')


if __name__ == '__main__':
    run()
