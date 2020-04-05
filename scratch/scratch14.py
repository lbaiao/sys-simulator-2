#  Testing for script 23

import sys
import os

lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from general import general as gen
from devices.devices import node, base_station, mobile_user, d2d_user, d2d_node_type
from pathloss import pathloss
from plots.plots import plot_positions, plot_spectral_effs
from q_learning.environments.completeEnvironment2 import CompleteEnvironment2
from dqn.agents.dqnAgent import ExternalDQNAgent
from dqn.externalDQNFramework import ExternalDQNFramework
from q_learning.q_table import DistributedQTable
from q_learning import rewards
from parameters.parameters import EnvironmentParameters, TrainingParameters, DQNAgentParameters, LearningParameters
from typing import List
from matplotlib import pyplot as plt

import torch
import math
import numpy as np
import os

def test(env: CompleteEnvironment2, framework: ExternalDQNFramework, max_d2d: int, num_episodes: int, episode_steps: int):
    mue_spectral_effs = [list() for i in range(max_d2d+1)]
    d2d_spectral_effs = [list() for i in range(max_d2d+1)]   
    done = False
    bag = list()
    aux_range = range(max_d2d)[1:]
    for _ in range(num_episodes):
        actions = [i*0.82*p_max/5/1000 for i in range(5)] # best result
        n_agents = np.random.choice(aux_range)
        agents = [ExternalDQNAgent(agent_params, actions) for i in range(n_agents)] # 1 agent per d2d tx        
        env.build_scenario(agents)
        done = False
        obs = [env.get_state(a) for a in agents] 
        total_reward = 0.0
        i = 0
        while not done:                        
            for j, agent in enumerate(agents):
                    aux = agent.act(framework, obs[j]).max(1)
                    agent.set_action(aux[1].long(), agent.actions[aux[1]])
                    bag.append(aux[1].item())
            next_obs, rewards, done = env.step(agents)
            obs = next_obs
            total_reward += sum(rewards)
            i +=1
            if i >= episode_steps:
                break
        mue_spectral_effs[n_agents].append(env.mue_spectral_eff.item())
        d2d_spectral_effs[n_agents].append(env.d2d_spectral_eff.item())
    return total_reward, mue_spectral_effs, d2d_spectral_effs, bag

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
# MAX_NUM_EPISODES = 2500
# MAX_NUM_EPISODES = 8000
# MAX_NUM_EPISODES = int(1.2e4)
# MAX_NUM_EPISODES = int(6e3)
STEPS_PER_EPISODE = 4000
# STEPS_PER_EPISODE = 200
# STEPS_PER_EPISODE = 1000
EPSILON_MIN = 0.01
# MAX_NUM_STEPS = 50
# EPSILON_DECAY = 4e-2 *  EPSILON_MIN / STEPS_PER_EPISODE
EPSILON_DECAY = 100 * EPSILON_MIN / STEPS_PER_EPISODE
MAX_NUM_EPISODES = int(1.2/EPSILON_DECAY)
# EPSILON_DECAY = 8e-1 *  EPSILON_MIN / STEPS_PER_EPISODE
# EPSILON_DECAY = 2 *  EPSILON_MIN / MAX_NUM_STEPS
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
# C = 8000 # C constant for the improved reward function
C = 80 # C constant for the improved reward function
TARGET_UPDATE = 10
MAX_NUMBER_OF_AGENTS = 10
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
environment = CompleteEnvironment2(env_params, reward_function, early_stop=1e-6, tolerance=10)

framework = ExternalDQNFramework(agent_params)
framework.policy_net.load_state_dict(torch.load(f'/home/lucas/dev/sys-simulator-2/models/script23.pt'))

reward_function = rewards.dis_reward_tensor

# policy 5 test
total_reward, mue_spectral_effs, d2d_spectral_effs, bag = test(environment, framework, MAX_NUMBER_OF_AGENTS, 5000, 25)

mue_success_rate = list()
for i, m in enumerate(mue_spectral_effs):    
    mue_success_rate.append(np.average(m > np.log2(1 + sinr_threshold_mue)))

d2d_speffs_avg = list()
for i, d in enumerate(d2d_spectral_effs):    
    d2d_speffs_avg.append(np.average(d))

log = list()
for i, d in enumerate(zip(d2d_speffs_avg, mue_success_rate)):
    log.append(f'NUMBER OF D2D_USERS: {i+1}')
    log.append(f'D2D SPECTRAL EFFICIENCY - SCRIPT: {d[0]}')
    log.append(f'MUE SUCCESS RATE - SCRIPT: {d[1]}')
    log.append(f'-------------------------------------')


filename = gen.path_leaf(__file__)
filename = filename.split('.')[0]
filename = f'{lucas_path}/logs/{filename}.txt'
file = open(filename, 'w')

for l in log:
    file.write(f'{l}\n')
file.close()

plt.figure(1)
plt.hist(bag)
plt.xlabel('Actions')
plt.ylabel('Number of occurrences')

fig2, ax1 = plt.subplots()
ax1.set_xlabel('Number of D2D pairs in the RB')
ax1.set_ylabel('D2D Average Spectral Efficiency [bps/Hz]', color='tab:blue')
ax1.plot(d2d_speffs_avg, '.', color='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('MUE Success Rate', color='tab:red')
ax2.plot(mue_success_rate, '.', color='tab:red')
fig2.tight_layout()


# plt.figure(1)
# plt.plot(list(range(len(d2d_spectral_effs))), d2d_spectral_effs, '.', label='Script')
# plt.title('D2D spectral efficiencies')
# plt.legend()

# plt.figure(2)
# threshold_eff = np.log2(1 + sinr_threshold_mue) * np.ones(len(mue_spectral_effs))
# plt.plot(list(range(len(mue_spectral_effs))), mue_spectral_effs, '.', label='Script ')
# plt.plot(list(range(len(mue_spectral_effs))), threshold_eff, label='Threshold')    

# plt.title('MUE spectral efficiencies')
# plt.legend()


plt.show()









