## script 12 test

import sys

import os
lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from q_learning.environments.distanceEnvironment import DistanceEnvironment
from q_learning.agents.distanceAgent import DistanceAgent
from q_learning.rewards import dis_reward, centralized_reward
from parameters.parameters import EnvironmentParameters, TrainingParameters, AgentParameters, LearningParameters
from general import general as gen
from plots.plots import plot_spectral_effs
from typing import List
from matplotlib import pyplot as plt

import numpy as np


def test(agents: List[DistanceAgent], env: DistanceEnvironment, policies: np.array, num_episodes: int, episode_steps: int):
    mue_spectral_effs = list()
    d2d_spectral_effs = list()    
    done = False
    for _ in range(num_episodes):
        env.build_scenario(agents)
        done = False
        obs = [env.get_state(a) for a in agents] 
        total_reward = 0.0
        i = 0
        while not done:            
            actions_indexes = np.zeros(len(agents), dtype=int)        
            for m in range(len(agents)):
                actions_indexes[m] = policies[m][obs[m]]
                agents[m].set_action(actions_indexes[m])
            next_obs, reward, done = env.step(agents)
            obs = next_obs
            total_reward += reward
            i +=1
            if i >= episode_steps:
                break
        mue_spectral_effs.append(env.mue_spectral_eff)
        d2d_spectral_effs.append(env.d2d_spectral_eff)
    return total_reward, mue_spectral_effs, d2d_spectral_effs

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
sinr_threshold = 6  # mue sinr threshold in dB
mue_margin = .5e4

# conversions from dB to pow
p_max = p_max - 30
p_max = gen.db_to_power(p_max)
noise_power = noise_power - 30
noise_power = gen.db_to_power(noise_power)
bs_gain = gen.db_to_power(bs_gain)
user_gain = gen.db_to_power(user_gain)
sinr_threshold = gen.db_to_power(sinr_threshold)

# q-learning parameters
# MAX_NUM_EPISODES = 1e5
MAX_NUM_EPISODES = 1000
# STEPS_PER_EPISODE = 400
STEPS_PER_EPISODE = 200 
EPSILON_MIN = 0.05
# max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
# MAX_NUM_STEPS = 50
# EPSILON_DECAY = 4e-2 *  EPSILON_MIN / STEPS_PER_EPISODE
EPSILON_DECAY = 5e-1 *  EPSILON_MIN / STEPS_PER_EPISODE
# EPSILON_DECAY = 2 *  EPSILON_MIN / MAX_NUM_STEPS
ALPHA = 0.5  # Learning rate
GAMMA = 0.9  # Discount factor
C = 80  # C constant for the improved reward function

# more parameters
env_params = EnvironmentParameters(rb_bandwidth, d2d_pair_distance, p_max, noise_power, bs_gain, user_gain, sinr_threshold,
                                        n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin)
train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
agent_params = AgentParameters(EPSILON_MIN, EPSILON_DECAY, 1)
learn_params = LearningParameters(ALPHA, GAMMA)

actions = [i*p_max/10/1000 + 1e-9 for i in range(11)]
reward_function10 = dis_reward
environment10 = DistanceEnvironment(env_params, reward_function10)
learned_policy_10 = np.load(f'{lucas_path}/models/script12.npy')


# policy 10 test
t_agents = [DistanceAgent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
total_reward, mue_spectral_effs10, d2d_spectral_effs10 = test(t_agents, environment10, learned_policy_10, 1000, 50)


mue_spectral_effs10 = np.array(mue_spectral_effs10)
mue_spectral_effs10 = np.reshape(mue_spectral_effs10, np.prod(mue_spectral_effs10.shape))

d2d_spectral_effs10 = np.array(d2d_spectral_effs10)
d2d_spectral_effs10 = np.reshape(d2d_spectral_effs10, np.prod(d2d_spectral_effs10.shape))

d2d_speffs_avg10 = np.average(d2d_spectral_effs10)

mue_success_rate10 = np.average(mue_spectral_effs10 > sinr_threshold)

log = list()
log.append(f'D2D SPECTRAL EFFICIENCY - SCRIPT: {d2d_speffs_avg10}')
log.append(f'MUE SUCCESS RATE - SCRIPT: {mue_success_rate10}')

filename = gen.path_leaf(__file__)
filename = filename.split('.')[0]
filename = f'{lucas_path}/logs/{filename}.txt'
file = open(filename, 'w')
for l in log:
    file.write(f'{l}\n')
file.close()

