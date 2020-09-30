# Similar to scratch3, but with the BAN channel
from sys_simulator.channels import BANChannel
from sys_simulator.general import general as gen
from sys_simulator.pathloss import pathloss_bs_users
from sys_simulator.plots import plot_positions_actions_pie
from sys_simulator.q_learning.environments.completeEnvironment5 \
    import CompleteEnvironment5
from sys_simulator.q_learning.agents.agent import Agent
from sys_simulator.q_learning.rewards import dis_reward_tensor2
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
STEPS_PER_EPISODE = 4000
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
actions = torch.tensor([i*0.82*p_max/5/1000 for i in range(5)])
channel = BANChannel()
env = CompleteEnvironment5(env_params, dis_reward_tensor2, channel)
pairs_positions = [
    (250, 0),
    (-250, 0),
    (0, 250),
    (0, -250)
]
mue_position = (500 / math.sqrt(2), 500 / math.sqrt(2))
tx_powers_indexes = [
    4, 4, 4, 4
]
# actions = [i*0.82*p_max/5/1000 for i in range(5)]  # best result
actions = [i for i in range(5)]  # best result
n_agents = len(pairs_positions)


def run():
    if len(pairs_positions) != len(tx_powers_indexes):
        raise Exception(
            'Different `pair_positions` and `tx_powers_indexes` lengths.'
        )
    agents = [Agent(agent_params, actions)
              for _ in range(n_agents)]  # 1 agent per d2d tx
    env.set_scenario(pairs_positions, mue_position, agents)
    obs = [env.get_state(a) for a in agents]
    total_reward = 0.0
    for j, agent in enumerate(agents):
        agent.set_action(tx_powers_indexes[j])
    next_obs, rewards, _ = env.step(agents)
    obs = next_obs
    total_reward += sum(rewards)
    d2d_txs, d2d_rxs = zip(*env.d2d_pairs)
    # D2D interference on the MUE
    d2d_interferences = [
        d.tx_power * env.params.user_gain * env.params.bs_gain /
        pathloss_bs_users(d.distance_to_bs/1000) for d in d2d_txs
    ]
    d2d_total_interference = np.sum(d2d_interferences)
    percentage_interferences = d2d_interferences / d2d_total_interference
    if d2d_total_interference != 0:
        plot_positions_actions_pie(
            env.bs, env.mue, d2d_txs, d2d_rxs,
            tx_powers_indexes, percentage_interferences,
            obs[0][0][4].item(), sinr_threshold_mue,
            env.reward.item()
        )
    # show plots
    plt.show()