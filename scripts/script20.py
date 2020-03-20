# Classic q-learning solution but with d2d users varying from 1 to 20
# state is based on number of d2d pairs, agent distance to bs, interference indicator

import sys
import os

lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from general import general as gen
from devices.devices import node, base_station, mobile_user, d2d_user, d2d_node_type
from pathloss import pathloss
from plots.plots import plot_positions, plot_spectral_effs
from q_learning.environments.distanceEnvironmentMulti import DistanceEnvironmentMulti
from q_learning.agents.distanceAgent import DistanceAgent
from q_learning.q_table import QTensor
from q_learning import rewards
from parameters.parameters import EnvironmentParameters, TrainingParameters, AgentParameters, LearningParameters
from typing import List
from matplotlib import pyplot as plt

import math
import numpy as np

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
# MAX_NUM_EPISODES = 2500
# MAX_NUM_EPISODES = 8000
STEPS_PER_EPISODE = 200
# STEPS_PER_EPISODE = 1000
EPSILON_MIN = 0.05
# MAX_NUM_STEPS = 50
# EPSILON_DECAY = 4e-2 *  EPSILON_MIN / STEPS_PER_EPISODE
# EPSILON_DECAY = 2e-2 *  EPSILON_MIN / STEPS_PER_EPISODE
# EPSILON_DECAY = 8e-1 *  EPSILON_MIN / STEPS_PER_EPISODE
EPSILON_DECAY = .16e-4
MAX_NUM_EPISODES = 100000
# EPSILON_DECAY = 2 *  EPSILON_MIN / MAX_NUM_STEPS
ALPHA = 0.2  # Learning rate
GAMMA = 0.98  # Discount factor
C = 80  # C constant for the improved reward function
MAX_NUMBER_OF_AGENTS = 20

# more parameters
env_params = EnvironmentParameters(rb_bandwidth, d2d_pair_distance, p_max, noise_power, bs_gain, user_gain, sinr_threshold,
                                        n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin)
train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
agent_params = AgentParameters(EPSILON_MIN, EPSILON_DECAY, 1)
learn_params = LearningParameters(ALPHA, GAMMA)

actions = [i*0.82*p_max/5/1000 for i in range(5)] # best result
agents = [DistanceAgent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
# q_tables = [QTable(len(actions)*2, len(actions), learn_params) for a in agents]
q_table = QTensor(np.zeros([MAX_NUMBER_OF_AGENTS, len(actions), 2, len(actions)]), learn_params)
reward_function = rewards.centralized_reward
environment = DistanceEnvironmentMulti(env_params, reward_function)

# training function
# TODO: colocar agente e d2d_device na mesma classe? fazer propriedade d2d_device no agente?
def train(env: DistanceEnvironmentMulti, params: TrainingParameters, q_table: QTensor, max_d2d: int):
    best_reward = float('-inf')
    aux_range = range(max_d2d)[1:]
    epsilon = agent_params.start_epsilon
    rewards = list()
    avg_q_values = list()
    for episode in range(params.max_episodes):
        actions = [i*0.82*p_max/5/1000 for i in range(5)] # best result
        n_agents = np.random.choice(aux_range)
        env.params.n_d2d = n_agents
        agents = [DistanceAgent(agent_params, actions) for i in range(n_agents)] # 1 agent per d2d tx
        env.build_scenario(agents)
        counts = np.zeros(n_agents)
        awaits = list()
        await_steps = [2,3,4]
        for a in agents:
            awaits.append(np.random.choice(await_steps))
            a.set_action(0)
            a.set_epsilon(epsilon)

        done = False
        obs = [env.get_state(a) for a in agents]                
        total_reward = 0.0
        i = 0
        while not done:
            if i >= params.steps_per_episode:
                break
            else:
                for j in range(n_agents):
                    if counts[j] < awaits[j]:
                        counts[j] += 1
                    else:
                        agents[j].get_action(obs[j], q_table)                
                        counts[j] = 0
                        awaits[j] = np.random.choice(await_steps)
                next_obs, interference_indicator, reward, done = env.step(agents)                
                i += 1
                for m, a in enumerate(agents):                    
                    q_table.learn((n_agents, obs[m], interference_indicator), a.action_index, reward, (n_agents, next_obs[m], interference_indicator))
                obs = next_obs
                total_reward += reward
                # total_reward += sum(reward)
            if total_reward > best_reward:
                best_reward = total_reward
            avg_q_values.append(np.mean(q_table.tensor))
            print("Episode#:{} sum reward:{} best_sum_reward:{} eps:{}".format(episode,
                                     total_reward, best_reward, agents[0].epsilon))
        rewards.append(total_reward)        
        epsilon = agents[0].epsilon

    # Return the trained policy
    policy = np.argmax(q_table, axis=len(q_table.shape)-1)
    return policy, rewards, avg_q_values


def test(agents: List[DistanceAgent], env: DistanceEnvironmentMulti, policies: np.array, num_episodes: int, episode_steps: int):
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

            
# SCRIPT EXEC
# training
learned_policies, train_rewards, avg_q_values = train(environment, train_params, q_table, MAX_NUMBER_OF_AGENTS)

filename = gen.path_leaf(__file__) 
filename =  filename.split('.')[0]
np.save(f'{lucas_path}/models/{filename}', learned_policies)

# testing
t_env = DistanceEnvironmentMulti(env_params, reward_function)
t_agents = [DistanceAgent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
total_reward, mue_spectral_effs, d2d_spectral_effs = test(t_agents, environment, learned_policies, 5, 100)

# plots
mue_spectral_effs = np.array(mue_spectral_effs)
mue_spectral_effs = np.reshape(mue_spectral_effs, np.prod(mue_spectral_effs.shape))

d2d_spectral_effs = np.array(d2d_spectral_effs)
d2d_spectral_effs = np.reshape(d2d_spectral_effs, np.prod(d2d_spectral_effs.shape))

threshold_eff = np.log2(1 + sinr_threshold) * np.ones(len(mue_spectral_effs))

# plt.figure(1)
# plt.plot(list(range(len(d2d_spectral_effs))), d2d_spectral_effs, '.',label='D2D')
# plt.plot(list(range(len(mue_spectral_effs))), mue_spectral_effs, '.',label='MUE')
# plt.plot(list(range(len(mue_spectral_effs))), threshold_eff, label='Threshold')    
# plt.title('Spectral efficiencies')
# plt.legend()


# normalized_reward = (train_rewards - np.mean(train_rewards))/np.std(train_rewards)

# plt.figure(2)
# plt.plot(normalized_reward, '.')
# plt.title('Total rewards')

plottable = avg_q_values[::10]

plt.figure(3)
plt.plot(range(len(plottable)), plottable, '.')
plt.xlabel('Iterations*1/10')
plt.ylabel('Average Q-Values')


plt.show()


