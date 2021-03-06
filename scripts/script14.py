# Simulation implemented for the Distributed-Q Learning Based Power Control algorithm based in the algorithms found on  
#     Nie, S., Fan, Z., Zhao, M., Gu, X. and Zhang, L., 2016, September. Q-learning based power control algorithm for D2D communication. 
#     In 2016 IEEE 27th Annual International Symposium on Personal, Indoor, and Mobile Radio Communications 
#     (PIMRC) (pp. 1-6). IEEE.
#  In this simulation, the agent state is based on its position and the MUE sinr. The reward function is the Distributed Reward.
#  The reinforcement learning algorithm is Deep Q Learning. The agents train the same DQN

import sys
import os

lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from general import general as gen
from devices.devices import node, base_station, mobile_user, d2d_user, d2d_node_type
from pathloss import pathloss
from plots.plots import plot_positions, plot_spectral_effs
from q_learning.environments.completeEnvironment import CompleteEnvironment
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
EPSILON_DECAY = 3.236*1e-4    # medium training
# EPSILON_DECAY = 8.09*1e-4      # short training
# MAX_NUM_EPISODES = 40000      # super long training
# MAX_NUM_EPISODES = 20000      # long training
MAX_NUM_EPISODES = 5000      # medium training
# MAX_NUM_EPISODES = 2000        # short training
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
# C = 8000 # C constant for the improved reward function
C = 80 # C constant for the improved reward function
TARGET_UPDATE = 10

# more parameters
env_params = EnvironmentParameters(rb_bandwidth, d2d_pair_distance, p_max, noise_power, bs_gain, user_gain, sinr_threshold_train,
                                        n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin)
train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
agent_params = DQNAgentParameters(EPSILON_MIN, EPSILON_DECAY, 1, 512, GAMMA)

extFramework = ExternalDQNFramework(agent_params)
# actions = [i*p_max/10/1000 for i in range(21)] # worst
# actions = [i*0.80*p_max/10/1000 for i in range(21)] # best histogram
actions = [i*0.82*p_max/10/1000 for i in range(21)] # best result
agents = [ExternalDQNAgent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
reward_function = rewards.dis_reward_tensor2
environment = CompleteEnvironment(env_params, reward_function, early_stop=1e-6, tolerance=10)


# training function
# TODO: colocar agente e d2d_device na mesma classe? fazer propriedade d2d_device no agente?
def train(agents: List[ExternalDQNAgent], framework: ExternalDQNFramework, env: CompleteEnvironment, params: TrainingParameters):
    counts = np.zeros(len(agents))
    awaits = list()
    await_steps = [2,3,4]
    for a in agents:
        awaits.append(np.random.choice(await_steps))
        a.set_action(torch.tensor(0).long().cuda(), a.actions[0])
    best_reward = 1e-16
    device = torch.device('cuda')
    rewards_bag = list()
    for episode in range(params.max_episodes):
        # TODO: atualmente redistribuo os usuarios aleatoriamente a cada episodio. Isto é o melhor há se fazer? 
        # Simular deslocamento dos usuários?
        env.build_scenario(agents)
        done = False
        obs = [env.get_state(a) for a in agents] 
        total_reward = 0.0
        i = 0
        bag = list()
        while not done:
            if i >= params.steps_per_episode:
                break
            else:
                actions = torch.zeros([len(agents)], device=device)
                for j, agent in enumerate(agents):
                    if counts[j] < awaits[j]:
                        counts[j] += 1
                    else:
                        agent.get_action(framework, obs[j])
                        actions[j] = agent.action_index                
                        counts[j] = 0
                        awaits[j] = np.random.choice(await_steps)
                # for j, agent in enumerate(agents):
                #     actions[j] = agent.get_action(obs[j])       
                next_obs, rewards, done = env.step(agents)                
                i += 1
                for j, agent in enumerate(agents):
                    framework.replay_memory.push(obs[j], actions[j], next_obs[j], rewards[j])
                    framework.learn()
                obs = next_obs
                total_reward += torch.sum(rewards)      
                bag.append(total_reward.item())      
                obs = next_obs
                if episode % TARGET_UPDATE == 0:
                    framework.target_net.load_state_dict(framework.policy_net.state_dict())
            if total_reward > best_reward:
                best_reward = total_reward
            print("Episode#:{} sum reward:{} best_sum_reward:{} eps:{}".format(episode,
                                     total_reward, best_reward, agents[0].epsilon))
        rewards_bag.append(np.average(bag))
        
    
    # Return the trained policy
    # policies = [np.argmax(q.table, axis=1) for q in q_tables]
    return rewards_bag

            
# SCRIPT EXEC
# training
# train(agents, environment, train_params)
rewards = train(agents, extFramework, environment, train_params)
# rewards = rb_bandwidth*rewards

cwd = os.getcwd()

torch.save(extFramework.policy_net.state_dict(), f'{cwd}/models/ext_model_dqn_agent.pt')
filename = gen.path_leaf(__file__)
filename = filename.split('.')[0]
filename = f'{lucas_path}/data/{filename}.pickle'
with open(filename, 'wb') as f:
    pickle.dump(extFramework.bag, f)

plt.figure(1)
plt.plot(extFramework.bag, '.')
plt.xlabel('Iterations')
plt.ylabel('Average Q-Values')

plt.figure(2)
plt.plot(rewards, '.')

plt.show()

# filename = gen.path_leaf(__file__) 
# filename =  filename.split('.')[0]
# np.save(f'{lucas_path}/models/{filename}', learned_policies)

# testing
# t_env = CompleteEnvironment(env_params, reward_function)
# t_agents = [DQNAgent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
# total_reward, mue_spectral_effs, d2d_spectral_effs = test(t_agents, environment, learned_policies, 5, 100)

# plots
# mue_spectral_effs = np.array(mue_spectral_effs)
# mue_spectral_effs = np.reshape(mue_spectral_effs, np.prod(mue_spectral_effs.shape))

# d2d_spectral_effs = np.array(d2d_spectral_effs)
# d2d_spectral_effs = np.reshape(d2d_spectral_effs, np.prod(d2d_spectral_effs.shape))

# threshold_eff = np.log2(1 + sinr_threshold_train) * np.ones(len(mue_spectral_effs))

# plt.figure(1)
# plt.plot(list(range(len(d2d_spectral_effs))), d2d_spectral_effs, '.',label='D2D')
# plt.plot(list(range(len(mue_spectral_effs))), mue_spectral_effs, '.',label='MUE')
# plt.plot(list(range(len(mue_spectral_effs))), threshold_eff, label='Threshold')    
# plt.title('Spectral efficiencies')
# plt.legend()
# plt.show()


