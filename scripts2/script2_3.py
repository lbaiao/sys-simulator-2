# same as script 14, but with the smaller dqn and returning the average q values. The devices positions are fixed.
# it uses a simpler environment


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
EPSILON_DECAY = 8.09*1e-4      # short training
# EPSILON_DECAY = 4.045*1e-4      # short training
# MAX_NUM_EPISODES = 40000      # super long training
# MAX_NUM_EPISODES = 20000      # long training
# MAX_NUM_EPISODES = 5000      # medium training
MAX_NUM_EPISODES = 2000        # short training
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
# C = 8000 # C constant for the improved reward function
C = 80 # C constant for the improved reward function
TARGET_UPDATE = 10
REPLAY_MEMORY_SIZE = 32
BATCH_SIZE = 16

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
    rewards_bag = list()
    env.build_scenario(agents)
    obs = env.get_state()
    total_reward = 0.0
    i = 0
    bag = list()
    for episode in range(params.max_episodes):
        actions = torch.zeros([len(agents)], device=device)
        for j, agent in enumerate(agents):
            agent.get_action(frameworks[j], obs)            
        next_obs, rewards = env.step(agents)                
        i += 1
        for j, agent in enumerate(agents):
            frameworks[j].replay_memory.push(obs, actions[j], next_obs, rewards[j])
            frameworks[j].learn()
        obs = next_obs
        total_reward += torch.sum(rewards)      
        bag.append(total_reward.item())      
        obs = next_obs
        if episode % TARGET_UPDATE == 0:
            for f in frameworks:
                f.target_net.load_state_dict(f.policy_net.state_dict())
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode#:{} sum reward:{} best_sum_reward:{} eps:{}".format(episode,
                                    total_reward, best_reward, agents[0].epsilon))
        rewards_bag.append(np.average(bag))
    return rewards_bag

            
# SCRIPT EXEC
# training
# train(agents, environment, train_params)
rewards = train(agents, ext_frameworks, environment, train_params)
# rewards = rb_bandwidth*rewards

cwd = os.getcwd()

filename = gen.path_leaf(__file__)
filename = filename.split('.')[0]
filename_model = filename
filename = f'{lucas_path}/data/{filename}.pickle'
dir_path = f'{lucas_path}/models/{filename_model}'
os.mkdir(dir_path)
for i, f in enumerate(ext_frameworks):
    torch.save(f.policy_net.state_dict(), f'{dir_path}/model_{i}.pt')

# with open(filename, 'wb') as f:
#     pickle.dump(spectral_effs, f)

plt.figure(1)
plt.plot(ext_frameworks[0].bag, '.')
plt.xlabel('Iterations')
plt.ylabel('Average Q-Values')

plt.figure(2)
plt.plot(rewards, '.')

plt.show()


