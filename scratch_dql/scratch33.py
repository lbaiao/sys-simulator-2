# Simulations taking pictures of the users distributions and makes 
# pie graphs with the d2d interference contributions on the mue. 
# It uses script23 model.

import sys
import os

lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from general import general as gen
from devices.devices import node, base_station, mobile_user, d2d_user, d2d_node_type
from pathloss import pathloss
from plots.plots import plot_positions, plot_spectral_effs, plot_positions_actions_pie
from q_learning.environments.completeEnvironment2 import CompleteEnvironment2
from dqn.agents.dqnAgent import ExternalDQNAgent
from dqn.externalDQNFramework import ExternalDQNFramework
from q_learning.q_table import DistributedQTable
from q_learning import rewards
from parameters.parameters import EnvironmentParameters, TrainingParameters, DQNAgentParameters, LearningParameters
from typing import List
from matplotlib import pyplot as plt
from plots.plots import plot_positions_and_actions, pie_plot

import torch
import math
import numpy as np
import os
import pickle

def test(env: CompleteEnvironment2, framework: ExternalDQNFramework, episode_steps: int, aux_range: List[int]):
    done = False
    # aux_range = range(max_d2d+1)[1:]
    for n_agents in aux_range:
        actions = [i*0.82*p_max/5/1000 for i in range(5)] # best result
        agents = [ExternalDQNAgent(agent_params, actions) for i in range(n_agents)] # 1 agent per d2d tx        
        env.build_scenario(agents)
        done = False
        obs = [env.get_state(a) for a in agents] 
        total_reward = 0.0
        i = 0
        while not done:  
            actions_index = list()                     
            for j, agent in enumerate(agents):
                aux = agent.act(framework, obs[j]).max(1)
                actions_index.append(aux[1].item())
                agent.set_action(aux[1].long(), agent.actions[aux[1]])
            next_obs, rewards, done = env.step(agents)
            obs = next_obs
            total_reward += sum(rewards)
            i +=1
            if i >= episode_steps:
                break
        d2d_txs, d2d_rxs = zip(*env.d2d_pairs)
        d2d_interferences = [d.tx_power * env.params.user_gain * env.params.bs_gain / pathloss.pathloss_bs_users(d.distance_to_bs/1000) for d in d2d_txs]
        d2d_total_interference = np.sum(d2d_interferences)
        percentage_interferences = d2d_interferences / d2d_total_interference
        if d2d_total_interference != 0:
            plot_positions_actions_pie(env.bs, [env.mue], d2d_txs, d2d_rxs, actions_index, percentage_interferences, f'N={n_agents}', obs[0][0][4].item())
            # pie_plot(percentage_interferences, f'N={n_agents}')
        # plot_positions_and_actions(env.bs, [env.mue], d2d_txs, d2d_rxs, actions_index)
        

n_mues = 1 # number of mues
n_d2d = 2  # number of d2d pairs
n_rb = n_mues   # number of RBs
bs_radius = 500 #   bs radius in m

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
C = 80 # C constant for the improved reward function
TARGET_UPDATE = 10
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 128

# more parameters
cwd = os.getcwd()

env_params = EnvironmentParameters(rb_bandwidth, d2d_pair_distance, p_max, noise_power, bs_gain, user_gain, sinr_threshold_mue,
                                        n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin)
train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
agent_params = DQNAgentParameters(EPSILON_MIN, EPSILON_DECAY, 1, REPLAY_MEMORY_SIZE, BATCH_SIZE, GAMMA)

actions = torch.tensor([i*0.82*p_max/5/1000 for i in range(5)])
reward_function = rewards.dis_reward_tensor2
environment = CompleteEnvironment2(env_params, reward_function)

framework = ExternalDQNFramework(agent_params)
framework.policy_net.load_state_dict(torch.load(f'{lucas_path}/models/script30.pt'))

reward_function = rewards.dis_reward_tensor

# policy 5 test
aux_range = list(range(11))[1:]
test(environment, framework, 50, aux_range)
# total_reward, mue_spectral_effs, d2d_spectral_effs, bag, action_counts_total, equals_counts_total = test(environment, framework, MAX_NUMBER_OF_AGENTS, 10, 5, x)

plt.show()









