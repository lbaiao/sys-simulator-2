# Same as script 15, but there are only 5 actions options, hence the DQN has a smaller output layer.

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
STEPS_PER_EPISODE = 25
EPSILON_MIN = 0.05
# MAX_NUM_STEPS = 50
# EPSILON_DECAY = 0.4045*1e-4    # super long training
# EPSILON_DECAY = 0.809*1e-4    # long training
# EPSILON_DECAY = 0.809*1e-4    # medium training
EPSILON_DECAY = 3.35*1e-4    # medium training
# EPSILON_DECAY = 8.09*1e-4      # short training
# MAX_NUM_EPISODES = 40000      # super long training
# MAX_NUM_EPISODES = 20000      # long training
MAX_NUM_EPISODES = 480      # medium training
# MAX_NUM_EPISODES = 2000        # short training
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
# C = 8000 # C constant for the improved reward function
C = 80 # C constant for the improved reward function
TARGET_UPDATE = 10
MAX_NUMBER_OF_AGENTS = 20

# more parameters
env_params = EnvironmentParameters(rb_bandwidth, d2d_pair_distance, p_max, noise_power, bs_gain, user_gain, sinr_threshold_train,
                                        n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin)
train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
agent_params = DQNAgentParameters(EPSILON_MIN, EPSILON_DECAY, 1, 512, GAMMA)

ext_framework = ExternalDQNFramework(agent_params)
# actions = [i*p_max/10/1000 for i in range(21)] # worst
# actions = [i*0.80*p_max/10/1000 for i in range(21)] # best histogram
reward_function = rewards.dis_reward_tensor
# environment = CompleteEnvironment(env_params, reward_function, early_stop=1e-6, tolerance=10)
environment = CompleteEnvironment(env_params, reward_function)


# training function
# TODO: colocar agente e d2d_device na mesma classe? fazer propriedade d2d_device no agente?
def train(framework: ExternalDQNFramework, env: CompleteEnvironment, params: TrainingParameters, agent_params: DQNAgentParameters, max_d2d: int):    
    best_reward = float('-inf')
    device = torch.device('cuda')
    mue_spectral_eff_bag = list()
    d2d_spectral_eff_bag = list()
    aux_range = range(max_d2d)[1:]
    epsilon = agent_params.start_epsilon
    for episode in range(params.max_episodes):
        # TODO: atualmente redistribuo os usuarios aleatoriamente a cada episodio. Isto é o melhor há se fazer? 
        # Simular deslocamento dos usuários?
        actions = [i*0.82*p_max/5/1000 for i in range(5)] # best result
        n_agents = np.random.choice(aux_range)
        agents = [ExternalDQNAgent(agent_params, actions) for i in range(n_agents)] # 1 agent per d2d tx
        counts = np.zeros(len(agents))
        awaits = list()
        await_steps = [2,3,4]
        for a in agents:
            awaits.append(np.random.choice(await_steps))
            a.set_action(torch.tensor(0).long().cuda(), a.actions[0])
            a.set_epsilon(epsilon)

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

            # some statistics
            mue_spectral_eff_bag.append(env.mue_spectral_eff)     # mue spectral eff
            d2d_spectral_eff_bag.append(env.d2d_spectral_eff/env.params.n_d2d)   # average d2d spectral eff        
        epsilon = agents[0].epsilon

    
    # Return the trained policy
    return (mue_spectral_eff_bag, d2d_spectral_eff_bag)

            
# SCRIPT EXEC
# training
spectral_effs = train(ext_framework, environment, train_params, agent_params, MAX_NUMBER_OF_AGENTS)
mue_spectral_effs, d2d_spectral_effs = zip(*spectral_effs)

cwd = os.getcwd()

torch.save(ext_framework.policy_net.state_dict(), f'{cwd}/models/ext_model_dqn_agent_mult_small_dqn.pt')
filename = gen.path_leaf(__file__)
filename = filename.split('.')[0]
filename = f'{lucas_path}/data/{filename}.pickle'
with open(filename, 'wb') as f:
    pickle.dump(spectral_effs, f)

plt.figure(1)
plt.plot(mue_spectral_effs, '.', label='MUEs')
plt.plot(d2d_spectral_effs, '.', label='D2Ds')
plt.xlabel('Iteration')
plt.ylabel('Average Spectral Efficiencies')
plt.legend()

plt.show()


