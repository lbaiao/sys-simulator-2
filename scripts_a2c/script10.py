# Similar to script5. The same environment
# is passed from training to testing, after being reset.
# Uses CompleteEnvironment10dB.
# Simulates many times, for different number of agents, and take the averages.
# There are different channels to the BS and to the devices.
# Single episode convergence. Everything is in dB. One NN for each agent.
from torch import nn
from sys_simulator.a2c.agent import Agent
from torch import optim
from sys_simulator.a2c import ActorCriticDiscrete, compute_gae_returns
from time import time
from typing import List
from sys_simulator.general.general import db_to_power, power_to_db
from sys_simulator.channels import BANChannel, UrbanMacroLOSWinnerChannel
from sys_simulator.general import general as gen
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
# run params
# STEPS_PER_EPISODE = 1000
# TEST_STEPS_PER_EPISODE = 100
# MAX_NUM_EPISODES = 10
# debugging params
STEPS_PER_EPISODE = 2
TEST_STEPS_PER_EPISODE = 2
MAX_NUM_EPISODES = 10
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
ITERATIONS_PER_NUM_AGENTS = 30
# ITERATIONS_PER_NUM_AGENTS = 10
max_d2d = MAX_NUMBER_OF_AGENTS
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
    frameworks = [
        ActorCriticDiscrete(
            env_state_size,
            len(actions),
            HIDDEN_SIZE,
            NUM_HIDDEN_LAYERS
        )
        for _ in range(n_agents)
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actors_optimizers = [
        optim.Adam(a2c.actor.parameters(), lr=LEARNING_RATE)
        for a2c in frameworks
    ]
    critics_optimizers = [
        optim.Adam(a2c.critic.parameters(), lr=LEARNING_RATE)
        for a2c in frameworks
    ]
    agents = [Agent() for _ in range(n_agents)]
    env.build_scenario(agents)
    for _ in range(MAX_NUM_EPISODES):
        obs, _ = env.step(agents)
        log_probs = torch.zeros((n_agents, STEPS_PER_EPISODE)).to(device)
        values = torch.zeros((n_agents, STEPS_PER_EPISODE+1)).to(device)
        rewards = torch.zeros((n_agents, STEPS_PER_EPISODE)).to(device)
        entropy = torch.zeros((n_agents, STEPS_PER_EPISODE)).to(device)
        for i in range(STEPS_PER_EPISODE):
            for j, (agent, a2c) in enumerate(zip(agents, frameworks)):
                action_index, dist, value = agent.act_discrete(a2c, obs[j])
                agent.set_action(actions[action_index.item()])
                log_prob = dist.log_prob(action_index)
                log_probs[j][i] = log_prob
                values[j][i] = value
                entropy[j][i] = dist.entropy()
            # perform an environment step
            next_obs_t, rewards_t = env.step(agents)
            rewards[:, i] = torch.FloatTensor(rewards_t)
            obs = next_obs_t
        # gae and returns
        next_obs_t = torch.cat(obs, 0).to(device)
        for j, (agent, a2c) in enumerate(zip(agents, frameworks)):
            _, _, next_value_t = agent.act_discrete(a2c, next_obs_t[j])
            values[j][STEPS_PER_EPISODE] = next_value_t
        advantages, returns = compute_gae_returns(device, rewards, values)
        # update critics
        critics_losses = [
            nn.functional.mse_loss(v[:-1], r) for v, r in zip(values, returns)
        ]
        for loss, optimizer in zip(critics_losses, critics_optimizers):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # update actors
        actors_losses = torch.mul(advantages, log_probs)
        actors_losses -= BETA * entropy
        actors_losses = torch.sum(actors_losses, axis=1)
        for loss, optimizer in zip(actors_losses, actors_optimizers):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # Return the trained policy
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
        f.actor.eval()
        f.critic.eval()
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
        for j, (agent, framework) in enumerate(zip(agents, frameworks)):
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
    # debugging
    # if mue_success_rate < 1:
    #     print('bug')
    # jain_index_avg = list()
    # for i, j in enumerate(jain_index):
    #     jain_index_avg.append(np.average(j))
    # save data
    return mue_success_rate, mue_spectral_effs, d2d_spectral_effs, rewards


def run():
    mue_sucess_rate_total = []
    mue_spectral_effs_total = []
    d2d_spectral_effs_total = []
    rewards_total = []
    start = time()
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
            env = deepcopy(ref_env)
            frameworks, agents = train(n, env)
            mue_success_rate, mue_spectral_effs, d2d_spectral_effs, rewards = \
                test(env, frameworks, agents)
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
    data_path = f'data/a2c/{filename}.pickle'
    data = {
        'mue_success_rate': mue_sucess_rate_total,
        'd2d_speffs': d2d_spectral_effs_total,
        'mue_speffs': mue_spectral_effs_total,
        'rewards': rewards_total,
        'mue_sinr_threshold': sinr_threshold_train,
        'elapsed_time': now
    }
    with open(data_path, 'wb') as file:
        pickle.dump(data, file)
    now = (time() - start) / 60
    print(f'done. Elapsed time: {now} minutes.')


if __name__ == '__main__':
    run()