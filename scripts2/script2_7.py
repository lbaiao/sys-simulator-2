# Simulation implemented for the Distributed-Q Learning Based Power Control algorithm found in 
#     Nie, S., Fan, Z., Zhao, M., Gu, X. and Zhang, L., 2016, September. Q-learning based power control algorithm for D2D communication. 
#     In 2016 IEEE 27th Annual International Symposium on Personal, Indoor, and Mobile Radio Communications 
#     (PIMRC) (pp. 1-6). IEEE.
# devices positions are fixed. Same as script 2_1, but we simulate for N = 2, 5, 10, and take the learning curves.

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
import numpy as np
import matplotlib.pyplot as plt

n_mues = 1 # number of mues
n_d2d_list = [2,5,10]  # number of d2d pairs
n_rb = n_mues   # number of RBs
bs_radius = 500 #   bs radius in m

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
# MAX_NUM_EPISODES = 1e5
MAX_NUM_EPISODES = 3000
# STEPS_PER_EPISODE = 400
STEPS_PER_EPISODE = 200 
EPSILON_MIN = 0.05
# max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
# MAX_NUM_STEPS = 50
# EPSILON_DECAY = 4e-2 *  EPSILON_MIN / STEPS_PER_EPISODE
EPSILON_DECAY = 1.5/MAX_NUM_EPISODES
# EPSILON_DECAY = 2 *  EPSILON_MIN / MAX_NUM_STEPS
ALPHA = 0.2  # Learning rate
GAMMA = 0.98  # Discount factor
C = 80  # C constant for the improved reward function

# more parameters

train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
agent_params = AgentParameters(EPSILON_MIN, EPSILON_DECAY, 1)
learn_params = LearningParameters(ALPHA, GAMMA)
actions = [i*0.82*p_max/5/1000 for i in range(5)] # best result

reward_function = rewards.dis_reward



# training function
# TODO: colocar agente e d2d_device na mesma classe? fazer propriedade d2d_device no agente?
def train(agents: List[Agent], env: DistributedEnvironment, params: TrainingParameters, q_tables: List[DistributedQTable]):
    best_reward = -1e9
    env.build_scenario(agents)
    bag = list()
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
        bag.append(np.mean([np.mean(q.table) for q in q_tables]))
        print("Episode#:{} sum reward:{} best_sum_reward:{} eps:{}".format(episode,
                                    total_reward, best_reward, agents[0].epsilon))
    
    # Return the trained policy
    policies = [np.argmax(q.table, axis=1) for q in q_tables]
    return policies, bag


def test(agents: List[Agent], env: DistributedEnvironment, policies, iterations: int):
    # env.build_scenario(agents)
    env.mue_spectral_eff = list()
    env.d2d_spectral_eff = list()
    done = False
    obs = env.get_state()
    total_reward = 0.0
    i = 0
    while not done:        
        action_indexes = [policy[obs] for policy in policies]
        for j in range(len(agents)):
            agents[j].set_action(action_indexes[j])
        next_obs, rewards, done = env.step(agents)
        obs = next_obs
        total_reward += sum(rewards)
        i +=1
        if i >= iterations:
            break
    return total_reward
            

# SCRIPT EXEC
# training
learning_curves = list()
for n_d2d in n_d2d_list:
    env_params = EnvironmentParameters(rb_bandwidth, d2d_pair_distance, p_max, noise_power, bs_gain, user_gain, sinr_threshold,
                                   n_mues, n_d2d, n_rb, bs_radius, c_param=C)
    agents = [Agent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
    q_tables = [DistributedQTable(2, len(actions), learn_params) for a in agents]
    environment = DistributedEnvironment(env_params, reward_function) 
    learned_policies, avg_q_values = train(agents, environment, train_params, q_tables)
    learning_curves.append(avg_q_values)


filename = gen.path_leaf(__file__)
filename = filename.split('.')[0]
np.save(f'{lucas_path}/data/{filename}', learning_curves)

plt.figure()
for i, lc in enumerate(learning_curves):
    plt.plot(lc, label=f'N={n_d2d_list[i]}')
plt.xlabel('Iteration')
plt.ylabel('Average Q-Values')
plt.legend()
plt.show()


print('SUCCESS')
