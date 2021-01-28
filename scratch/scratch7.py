# Similar to scratch2, but everything is in dB
from sys_simulator.general import db_to_power
from sys_simulator.q_learning.rewards import dis_reward_tensor_db
from scipy.spatial.distance import euclidean
from sys_simulator.pathloss import pathloss_bs_users_db, pathloss_users_db
from sys_simulator.plots import plot_positions_actions_pie
from sys_simulator.q_learning.environments.completeEnvironment2dB \
    import CompleteEnvironment2dB
from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from sys_simulator.dqn.externalDQNFramework import ExternalDQNFramework
from sys_simulator.parameters.parameters \
    import EnvironmentParameters, TrainingParameters, DQNAgentParameters
from matplotlib import pyplot as plt
import os
import torch
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
mue_margin = 140  # mue sinr margin in dB
# conversions from dBm to dB
p_max = p_max - 30
noise_power = noise_power - 30
# q-learning parameters
STEPS_PER_EPISODE = 10
EPSILON_MIN = 0.01
EPSILON_DECAY = 100 * EPSILON_MIN / STEPS_PER_EPISODE
MAX_NUM_EPISODES = int(1.2/EPSILON_DECAY)
MAX_NUMBER_OF_AGENTS = 20
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
C = 80  # C constant for the improved reward function
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
env = CompleteEnvironment2dB(env_params, reward_function)
framework = ExternalDQNFramework(agent_params)
framework.policy_net.load_state_dict(
    torch.load(f'{cwd}/models/dql/script30.pt')
)
pairs_positions = [
    (250, 0),
    (-250, 0),
    (0, 250),
    (0, -250)
]
mue_position = (500 / math.sqrt(2), 500 / math.sqrt(2))
n_agents = len(pairs_positions)
episode_steps = STEPS_PER_EPISODE


def calculate_interferences(env: CompleteEnvironment2dB):
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
            if rx == env.bs:
                loss = pathloss_bs_users_db
            else:
                loss = pathloss_users_db
            interf = \
                tx.tx_power + tx.gain \
                - loss(euclidean(tx.position, rx.position)/1000) \
                + rx.gain
            interferences[i][j] = interf
    tx_labels = [d.id for d in txs]
    rx_labels = [d.id for d in rxs]
    return interferences, tx_labels, rx_labels


def run():
    actions = np.linspace(-100, p_max, 5)
    agents = [ExternalDQNAgent(agent_params, actions)
              for _ in range(n_agents)]  # 1 agent per d2d tx
    env.set_scenario(pairs_positions, mue_position, agents)
    obs = [env.get_state(a) for a in agents]
    total_reward = 0.0
    actions_index = list()
    for _ in range(episode_steps):
        actions_index = list()
        for j, agent in enumerate(agents):
            aux = agent.act(framework, obs[j].float()).max(1)
            actions_index.append(aux[1].item())
            agent.set_action(aux[1].long(), agent.actions[aux[1]])
        next_obs, rewards, _ = env.step(agents)
        obs = next_obs
        total_reward += sum(rewards)
    d2d_txs, d2d_rxs = zip(*env.d2d_pairs)
    # D2D interference on the MUE, in dB
    d2d_interferences = np.array([
        d.tx_power + env.params.user_gain + env.params.bs_gain -
        pathloss_bs_users_db(d.distance_to_bs/1000) for d in d2d_txs
    ])
    d2d_interferences_mag = db_to_power(d2d_interferences)
    d2d_total_interference = np.sum(d2d_interferences_mag)
    percentage_interferences = d2d_interferences_mag / d2d_total_interference
    interferences, tx_labels, rx_labels = calculate_interferences(env)
    if d2d_total_interference != 0:
        plot_positions_actions_pie(
            env.bs, env.mue, d2d_txs, d2d_rxs,
            actions_index, percentage_interferences,
            obs[0][0][4].item(), sinr_threshold_mue,
            env.reward.item(), interferences, tx_labels, rx_labels
        )
    plt.show()


if __name__ == '__main__':
    run()
