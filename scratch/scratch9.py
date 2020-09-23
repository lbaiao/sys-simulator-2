# Similar to scratch5, but everything is in dB
# Simulations taking pictures of the users distributions and makes
# pie graphs with the d2d interference contributions on the mue.
# It uses script23 model. In this scratch, the user chooses the
# nodes positions, and the D2D tx powers. There is no algorithm
# deciding the power levels.
from sys_simulator.q_learning.agents.agent import Agent
from sys_simulator.general.general import db_to_power
from sys_simulator.q_learning.rewards import dis_reward_tensor_db
from scipy.spatial.distance import euclidean
from sys_simulator.channels import BANChannel
from sys_simulator.plots import plot_positions_actions_pie
from sys_simulator.q_learning.environments.completeEnvironment5dB \
    import CompleteEnvironment5dB
from sys_simulator.parameters.parameters \
    import EnvironmentParameters, TrainingParameters, DQNAgentParameters
from matplotlib import pyplot as plt
import os
import numpy as np
import math


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
mue_margin = 200  # mue sinr margin in dB
# conversions from dBm to dB
p_max = p_max - 30
noise_power = noise_power - 30
# channel parameters
CHANNEL_RND = False
# q-learning parameters
STEPS_PER_EPISODE = 10
EPSILON_MIN = 0.01
EPSILON_DECAY = 100 * EPSILON_MIN / STEPS_PER_EPISODE
MAX_NUM_EPISODES = int(1.2/EPSILON_DECAY)
MAX_NUMBER_OF_AGENTS = 20
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
C = 5  # C constant for the improved reward function
TARGET_UPDATE = 10
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 128
# more parameters
cwd = os.getcwd()
# params objects
env_params = EnvironmentParameters(
    rb_bandwidth, d2d_pair_distance, p_max, noise_power,
    bs_gain, user_gain, sinr_threshold_mue, n_mues,
    n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin
)
train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
agent_params = DQNAgentParameters(
    EPSILON_MIN, EPSILON_DECAY, 1,
    REPLAY_MEMORY_SIZE, BATCH_SIZE, GAMMA
)
# actions, rewards, environment, agent
reward_function = dis_reward_tensor_db
channel = BANChannel(rnd=CHANNEL_RND)
env = CompleteEnvironment5dB(env_params, reward_function, channel)
pairs_positions = [
    (250, 0),
    (-250, 0),
    (0, 250),
    (0, -250)
]
tx_powers_indexes = [
    2, 2, 2, 2
]
mue_position = (250 / math.sqrt(2), 250 / math.sqrt(2))
n_agents = len(pairs_positions)
episode_steps = STEPS_PER_EPISODE


def calculate_interferences(env: CompleteEnvironment5dB):
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


def run():
    if len(pairs_positions) != len(tx_powers_indexes):
        raise Exception(
            'Different `pair_positions` and `tx_powers_indexes` lengths.'
        )
    actions = np.linspace(p_max-20, p_max, 5)
    actions[0] = -1000
    agents = [Agent(agent_params, actions)
              for _ in range(n_agents)]  # 1 agent per d2d tx
    env.set_scenario(pairs_positions, mue_position, agents)
    total_reward = 0.0
    for j, agent in enumerate(agents):
        agent.set_action(tx_powers_indexes[j])
    _, rewards, done = env.step(agents)
    total_reward += sum(rewards)
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
            tx_powers_indexes, percentage_interferences,
            not done, sinr_threshold_mue,
            env.reward.item(), interferences, tx_labels, rx_labels
        )
    print_stuff(actions, env)
    plt.show()


def print_stuff(actions, env: CompleteEnvironment5dB):
    actions = [f'{i:.2f}' for i in actions]
    sinr_d2ds = [f'{i:.2f}' for i in env.sinr_d2ds[0]]
    print(f'MUE Tx Power [dBW]: {env.mue.tx_power:.2f}')
    print(f'D2D Power levels [dBW]: {actions}')
    print(f'D2D SINR [dBW]: {sinr_d2ds}')
    print(f'D2D Spectral Efficiencies: {env.d2d_spectral_eff}')


if __name__ == '__main__':
    run()
