# Simulation implemented for the Team-Q Learning Based Power Control algorithm found in 
#     Nie, S., Fan, Z., Zhao, M., Gu, X. and Zhang, L., 2016, September. Q-learning based power control algorithm for D2D communication. 
#     In 2016 IEEE 27th Annual International Symposium on Personal, Indoor, and Mobile Radio Communications 
#     (PIMRC) (pp. 1-6). IEEE.

import sys
import os

lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from general import general as gen
from devices.devices import node, base_station, mobile_user, d2d_user, d2d_node_type
from pathloss import pathloss
from plots.plots import plot_positions, plot_spectral_effs
from q_learning.environments.environment import RLEnvironment
from q_learning.agents.agent import Agent
from q_learning.q_table import QTable
from q_learning import rewards
from parameters.parameters import EnvironmentParameters, TrainingParameters, AgentParameters, LearningParameters
from typing import List

import math
import numpy as np
from pprint import pprint

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

# conversions from dB to pow
p_max = p_max - 30
p_max = gen.db_to_power(p_max)
noise_power = noise_power - 30
noise_power = gen.db_to_power(noise_power)
bs_gain = gen.db_to_power(bs_gain)
user_gain = gen.db_to_power(user_gain)
sinr_threshold = gen.db_to_power(sinr_threshold)

# q-learning parameters
# MAX_NUM_EPISODES = 70
MAX_NUM_EPISODES = 1600
# STEPS_PER_EPISODE = 50
STEPS_PER_EPISODE = 20
EPSILON_MIN = 0.05
# max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODEfidtri
MAX_NUM_STEPS = 400
EPSILON_DECAY = 0.27 * EPSILON_MIN / MAX_NUM_STEPS
# EPSILON_DECAY = 2 *  EPSILON_MIN / MAX_NUM_STEPS
ALPHA = 0.5  # Learning rate
GAMMA = 0.9  # Discount factor
C = 80  # C constant for the improved reward function

# more parameters
env_params = EnvironmentParameters(rb_bandwidth, d2d_pair_distance, p_max, noise_power, bs_gain, user_gain, sinr_threshold,
                                        n_mues, n_d2d, n_rb, bs_radius)
train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
agent_params = AgentParameters(EPSILON_MIN, EPSILON_DECAY, 1)
learn_params = LearningParameters(ALPHA, GAMMA)

actions = [i*p_max/10 + 1e-9 for i in range(11)]
agents = [Agent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
q_table = QTable(2, len(actions), learn_params)
reward_function = rewards.centralized_reward
environment = RLEnvironment(env_params, reward_function, early_stop=1e-6, tolerance=10)

# training function
# TODO: colocar agente e d2d_device na mesma classe? fazer propriedade d2d_device no agente?
def train(agents: List[Agent], env: RLEnvironment, params: TrainingParameters, q_table: QTable):
    best_reward = -1e9
    for episode in range(params.max_episodes):
        # TODO: atualmente redistribuo os usuarios aleatoriamente a cada episodio. Isto é o melhor há se fazer? 
        # Simular deslocamento dos usuários?
        env.build_scenario(agents)
        done = False
        obs = env.get_state()
        total_reward = 0.0
        i = 0
        while not done:
            if i >= params.steps_per_episode:
                break
            else:
                for a in agents:
                    a.get_action(obs, q_table)
                next_obs, reward, done = env.step(agents)
                i += 1
                for a in agents:                    
                    q_table.learn(obs, a.action_index, reward, next_obs)
                obs = next_obs
                total_reward += reward
            if total_reward > best_reward:
                best_reward = total_reward
            print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode,
                                     total_reward, best_reward, agents[0].epsilon))
    
    # Return the trained policy
    policy = np.argmax(q_table.table, axis=1)
    return policy

def test(agents: List[Agent], env: RLEnvironment, policy, iterations: int):
    env.build_scenario(agents)
    done = False
    obs = env.get_state()
    total_reward = 0.0
    i = 0
    while not done:        
        action_index = policy[obs]
        for agent in agents:
            agent.set_action(action_index)
        next_obs, reward, done = env.step(agents)
        obs = next_obs
        total_reward += reward
        i +=1
        if i >= iterations:
            break
    return total_reward
            

# SCRIPT EXEC
# training
learned_policy = train(agents, environment, train_params, q_table)

filename = 'model1'
np.save(f'{lucas_path}/models/{filename}', learned_policy)

# testing
t_env = RLEnvironment(env_params, reward_function)
t_agents = [Agent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
for i in range(50):
    total_reward = test(t_agents, t_env, learned_policy, 20)
    print(f'TEST #{i} REWARD: {total_reward}')

plot_spectral_effs(environment)
plot_spectral_effs(t_env)

print('SUCCESS')

        







