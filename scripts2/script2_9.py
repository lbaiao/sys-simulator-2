# Simulation implemented for the Distributed-Q Learning Based Power Control algorithm found in 
#     Nie, S., Fan, Z., Zhao, M., Gu, X. and Zhang, L., 2016, September. Q-learning based power control algorithm for D2D communication. 
#     In 2016 IEEE 27th Annual International Symposium on Personal, Indoor, and Mobile Radio Communications 
#     (PIMRC) (pp. 1-6). IEEE.
# Same as script 2_6, but we vary the number of d2d_users between [1,10]. We measure the average usage of each action, and take
# the mean and variance, and save the data.

import sys
import os

lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from general import general as gen
from devices.devices import node, base_station, mobile_user, d2d_user, d2d_node_type
from pathloss import pathloss
from plots.plots import plot_positions, plot_spectral_effs
from q_learning.environments.distributedEnvironment import DistributedEnvironment
from q_learning.agents.agent import Agent
from q_learning.q_table import DistributedQTable
from q_learning import rewards
from parameters.parameters import EnvironmentParameters, TrainingParameters, AgentParameters, LearningParameters
from typing import List

import math
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt

n_mues = 1 # number of mues
n_rb = n_mues   # number of RBs
bs_radius = 500 #   bs radius in m
n_d2d_list = list(range(11))[1:]

rb_bandwidth = 180*1e3  # rb bandwidth in Hz
d2d_pair_distance = 50  # d2d pair distance in m
p_max = 23  # max tx power in dBm
noise_power = -116  # noise power per RB in dBm
bs_gain = 17    # macro bs antenna gain in dBi
user_gain = 4   # user antenna gain in dBi
sinr_threshold = 6  # mue sinr threshold in dB

# conversions from dB to pow
p_max = p_max - 30
p_max = gen.db_to_power(p_max)
noise_power = noise_power - 30
noise_power = gen.db_to_power(noise_power)
bs_gain = gen.db_to_power(bs_gain)
user_gain = gen.db_to_power(user_gain)
sinr_threshold = gen.db_to_power(sinr_threshold)

# q-learning parameters
MAX_NUM_EPISODES = 3000
ITERATIONS = 300
TEST_STEPS = 100
# MAX_NUM_EPISODES = 10
# ITERATIONS = 1
# TEST_STEPS = 100
STEPS_PER_EPISODE = 200 
EPSILON_MIN = 0.05
EPSILON_DECAY = 1/MAX_NUM_EPISODES
ALPHA = 0.2  # Learning rate
GAMMA = 0.98  # Discount factor
C = 80  # C constant for the improved reward function
MAX_NUM_AGENTS = 10

# training function
# TODO: colocar agente e d2d_device na mesma classe? fazer propriedade d2d_device no agente?
def train(agents: List[Agent], env: DistributedEnvironment, params: TrainingParameters, q_tables: List[DistributedQTable]):
    best_reward = -1e9
    env.build_scenario(agents)    
    for episode in range(params.max_episodes):
        # TODO: atualmente redistribuo os usuarios aleatoriamente a cada episodio. Isto é o melhor há se fazer? 
        # Simular deslocamento dos usuários?        
        obs = env.get_state()
        total_reward = 0.0
        for j in range(len(agents)):
            agents[j].get_action(obs, q_tables[j])                
        next_obs, rewards, _ = env.step(agents)
        for m in range(len(agents)):
            q_tables[m].learn(obs, agents[m].action_index, rewards[m], next_obs)
        obs = next_obs
        total_reward += sum(rewards)
        if total_reward > best_reward:
            best_reward = total_reward
        # print("Episode#:{} sum reward:{} best_sum_reward:{} eps:{}".format(episode,
        #                             total_reward, best_reward, agents[0].epsilon))
    
    # Return the trained policy
    policies = [np.argmax(q.table, axis=1) for q in q_tables]
    return policies


def test(agents: List[Agent], env: DistributedEnvironment, policies, iterations: int):
    # env.build_scenario(agents)
    env.mue_spectral_eff = list()
    env.d2d_spectral_eff = list()
    done = False
    obs = env.get_state()
    total_reward = 0.0
    i = 0
    aux = len(agents[0].actions)    
    action_counts = list()
    while not done:        
        action_indexes = [policy[obs] for policy in policies]
        action_counts.append([action_indexes.count(i) for i in range(aux)])        
        for j in range(len(agents)):
            agents[j].set_action(action_indexes[j])
        next_obs, rewards, done = env.step(agents)
        obs = next_obs
        total_reward += sum(rewards)
        i +=1
        if i >= iterations:
            break
    # action_counts = np.sum(action_counts, axis=0)
    return total_reward, env.mue_spectral_eff, env.d2d_spectral_eff, action_counts
            

mue_success_avg_total = list()
d2d_speffs_avg_total = list()
action_counts_total = list()
for n_d2d in n_d2d_list:
    env_params = EnvironmentParameters(rb_bandwidth, d2d_pair_distance, p_max, noise_power, bs_gain, user_gain, sinr_threshold,
                                            n_mues, n_d2d, n_rb, bs_radius, c_param=C)
    train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
    agent_params = AgentParameters(EPSILON_MIN, EPSILON_DECAY, 1)
    learn_params = LearningParameters(ALPHA, GAMMA)

    actions = [i*0.82*p_max/5/1000 for i in range(5)] # best result
    agents = [Agent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
    q_tables = [DistributedQTable(2, len(actions), learn_params) for a in agents]
    reward_function = rewards.dis_reward
    environment = DistributedEnvironment(env_params, reward_function)
    action_counts = list()

    # SCRIPT EXEC
    # training
    success_rates = list()
    mue_spectral_effs = list()
    d2d_spectral_effs = list()
    for i in range(ITERATIONS):
        print(f'Iteration {i+1}/{ITERATIONS}')
        learned_policies = train(agents, environment, train_params, q_tables)
        total_reward, mue_speffs, d2d_speffs, counts = test(agents, environment, learned_policies, TEST_STEPS)
        action_counts += counts
        success_rates.append(np.mean(np.array(mue_speffs) > sinr_threshold))
        mue_spectral_effs.append(np.mean(mue_speffs))
        d2d_spectral_effs.append(np.mean(d2d_speffs))
        
    success_rates_avg = np.mean(success_rates)
    mue_spectral_effs_avg = np.mean(mue_spectral_effs)
    d2d_spectral_effs_avg = np.mean(d2d_spectral_effs)
    mue_success_avg_total.append(success_rates_avg)
    d2d_speffs_avg_total.append(d2d_spectral_effs_avg)
    action_counts_total.append(action_counts)
    # action_counts_total.append(np.sum(action_counts, axis=0) / (n_d2d * ITERATIONS * TEST_STEPS))

    log = list()
    log.append(f'NUMBER OF D2D_USERS: {n_d2d}')
    log.append(f'D2D SPECTRAL EFFICIENCY: {d2d_spectral_effs_avg}')
    log.append(f'MUE SPECTRAL EFFICIENCY: {mue_spectral_effs_avg}')
    log.append(f'MUE SUCCESS RATE: {success_rates_avg}')
    log.append(f'-------------------------------------')


cwd = os.getcwd()

filename = gen.path_leaf(__file__)
filename = filename.split('.')[0]
pickle_filename = f'{lucas_path}/data/{filename}.pickle'

data = {
    'action_counts_total': action_counts_total,
    'd2d_speffs_avg_total': d2d_speffs_avg_total,
    'mue_success_avg_total': mue_success_avg_total
}

with open(pickle_filename, 'wb') as file:
    pickle.dump(data, file)

filename = f'{lucas_path}/logs/{filename}.txt'
file = open(filename, 'w')

for l in log:
    file.write(f'{l}\n')
file.close()

for (i, c) in zip(n_d2d_list, action_counts_total):
    plt.figure()
    plt.plot(np.mean(c, axis=0), '*',label='mean')
    plt.plot(np.var(c, axis=0), 'd', label='variance')
    plt.legend()
    plt.title(f'Actions Frequencies, N={i}')


fig2, ax1 = plt.subplots()
ax1.set_xlabel('Number of D2D pairs in the RB')
ax1.set_ylabel('D2D Average Spectral Efficiency [bps/Hz]', color='tab:blue')
ax1.plot(d2d_speffs_avg_total, '.', color='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('MUE Success Rate', color='tab:red')
ax2.plot(mue_success_avg_total, '.', color='tab:red')
fig2.tight_layout()

plt.show()
