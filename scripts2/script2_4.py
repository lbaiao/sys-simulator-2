# same as script2-3.py, but we train and test for many different scenarios and take the statistics.


import sys
import os

lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from general import general as gen
from devices.devices import node, base_station, mobile_user, d2d_user, d2d_node_type
from pathloss import pathloss
from plots.plots import plot_positions, plot_spectral_effs
from q_learning.environments.simpleEnvironment import SimpleEnvironment
from dqn.agents.dqnAgent import ExternalDQNAgent
from dqn.externalDQNFramework import ExternalDQNFramework
from dqn.replayMemory import ReplayMemory
from dqn.dqn import DQN
from q_learning.q_table import DistributedQTable
from q_learning import rewards
from parameters.parameters import EnvironmentParameters, TrainingParameters, DQNAgentParameters, LearningParameters
from typing import List
from matplotlib import pyplot as plt

import torch
import math
import numpy as np
import os
import pickle

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
sinr_threshold_train = 6  # mue sinr threshold in dB for training
sinr_threshold_mue = 6  # true mue sinr threshold in dB
mue_margin = .5e4

# conversions from dB to pow
p_max = p_max - 30
p_max = gen.db_to_power(p_max)
noise_power = noise_power - 30
noise_power = gen.db_to_power(noise_power)
bs_gain = gen.db_to_power(bs_gain)
user_gain = gen.db_to_power(user_gain)
sinr_threshold_train = gen.db_to_power(sinr_threshold_train)

# q-learning parameters
STEPS_PER_EPISODE = 100
EPSILON_MIN = 0.05
# MAX_NUM_STEPS = 50
# EPSILON_DECAY = 0.4045*1e-4    # super long training
# EPSILON_DECAY = 0.809*1e-4    # long training
# EPSILON_DECAY = 0.809*1e-4    # medium training
# EPSILON_DECAY = 3.236*1e-4    # medium training
# EPSILON_DECAY = 4.045*1e-4      # short training
# MAX_NUM_EPISODES = 40000      # super long training
# MAX_NUM_EPISODES = 20000      # long training
# MAX_NUM_EPISODES = 5000      # medium training
MAX_NUM_EPISODES = 10000        # short training
EPSILON_DECAY = 1.6/MAX_NUM_EPISODES      # short training
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
# C = 8000 # C constant for the improved reward function
C = 80 # C constant for the improved reward function
TARGET_UPDATE = 10
REPLAY_MEMORY_SIZE = 128
BATCH_SIZE = 32
ITERATIONS = 2

# more parameters
env_params = EnvironmentParameters(rb_bandwidth, d2d_pair_distance, p_max, noise_power, bs_gain, user_gain, sinr_threshold_train,
                                        n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin)
train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
agent_params = DQNAgentParameters(EPSILON_MIN, EPSILON_DECAY, 1, REPLAY_MEMORY_SIZE, BATCH_SIZE, GAMMA)

# actions = [i*p_max/10/1000 for i in range(21)] # worst
# actions = [i*0.80*p_max/10/1000 for i in range(21)] # best histogram
actions = [i*0.82*p_max/5/1000 for i in range(5)] # best result
ext_frameworks = [ExternalDQNFramework(agent_params) for _ in range(n_d2d)]
agents = [ExternalDQNAgent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
reward_function = rewards.dis_reward_tensor2
environment = SimpleEnvironment(env_params, reward_function)


# training function
# TODO: colocar agente e d2d_device na mesma classe? fazer propriedade d2d_device no agente?
def train(agents: List[ExternalDQNAgent], frameworks: List[ExternalDQNFramework], env: SimpleEnvironment, params: TrainingParameters):
    best_reward = 1e-16
    device = torch.device('cuda')
    env.build_scenario(agents)
    obs = env.get_state()
    total_reward = 0.0
    i = 0
    bag = list()
    for episode in range(params.max_episodes):
        actions = torch.zeros([len(agents)], device=device)
        for j, agent in enumerate(agents):
            agent.get_action(frameworks[j], obs) 
            actions[j] = agent.action_index           
        next_obs, rewards = env.step(agents)                
        i += 1
        for j, agent in enumerate(agents):
            frameworks[j].replay_memory.push(obs, actions[j], next_obs, rewards[j])
            frameworks[j].learn()
        obs = next_obs
        total_reward = torch.sum(rewards)      
        bag.append(total_reward.item())      
        obs = next_obs
        if episode % TARGET_UPDATE == 0:
            for f in frameworks:
                f.target_net.load_state_dict(f.policy_net.state_dict())
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode#:{} sum reward:{} best_sum_reward:{} eps:{}".format(episode,
                                    total_reward, best_reward, agents[0].epsilon))
    return bag


def test(env: SimpleEnvironment, agents: List[ExternalDQNAgent], frameworks: List[ExternalDQNFramework], num_episodes: int):
    bag = list()    
    obs = env.get_state()
    mue_spectral_effs = list()
    d2d_spectral_effs = list()
    for _ in range(num_episodes):
        total_reward = 0.0
        for i, agent in enumerate(agents):
            aux = agent.act(frameworks[i], obs).max(1)
            agent.set_action(aux[1].long(), agent.actions[aux[1]])
            bag.append(aux[1].item())
        next_obs, rewards = env.step(agents)
        obs = next_obs
        total_reward += sum(rewards)
        mue_spectral_effs.append(env.mue_spectral_eff.item())
        d2d_spectral_effs.append(env.d2d_spectral_eff.item())
    return total_reward, mue_spectral_effs, d2d_spectral_effs, bag

            
# SCRIPT EXEC
# training
success_rates = list()
mue_spectral_effs = list()
d2d_spectral_effs = list()
for _ in range(ITERATIONS):
    rewards = train(agents, ext_frameworks, environment, train_params)
    total_reward, mue_speffs, d2d_speffs, _ = test(environment, agents, ext_frameworks, 100)
    success_rates.append(np.mean(mue_speffs > sinr_threshold_mue))
    mue_spectral_effs.append(np.mean(mue_speffs))
    d2d_spectral_effs.append(np.mean(d2d_speffs))

success_rates_avg = np.mean(success_rates)
mue_spectral_effs_avg = np.mean(mue_spectral_effs)
d2d_spectral_effs_avg = np.mean(d2d_spectral_effs)

log = list()
log.append(f'NUMBER OF D2D_USERS: {n_d2d}')
log.append(f'D2D SPECTRAL EFFICIENCY: {d2d_spectral_effs_avg}')
log.append(f'MUE SPECTRAL EFFICIENCY: {mue_spectral_effs_avg}')
log.append(f'MUE SUCCESS RATE: {success_rates_avg}')
log.append(f'-------------------------------------')
    
# rewards = rb_bandwidth*rewards

cwd = os.getcwd()

filename = gen.path_leaf(__file__)
filename = filename.split('.')[0]
filename = f'{lucas_path}/logs/{filename}.txt'
file = open(filename, 'w')

for l in log:
    file.write(f'{l}\n')
file.close()


