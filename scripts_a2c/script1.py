# A2C script, but with completeEnvironment2. It uses multiple agents
# to train a single A2C network. The algorithm is trained with N_D2D
# varying from 1 to 10.

import sys
import os

lucas_path = os.getcwd()
sys.path.insert(1, lucas_path)

from general import general as gen
from devices.devices import node, base_station, \
    mobile_user, d2d_user, d2d_node_type
from pathloss import pathloss
from plots.plots import plot_positions, plot_spectral_effs
from q_learning.environments.completeEnvironmentA2C \
    import CompleteEnvironmentA2C
from q_learning import rewards
from parameters.parameters import EnvironmentParameters,\
    TrainingParameters, DQNAgentParameters, LearningParameters
from typing import List
from matplotlib import pyplot as plt

from a2c.agent import Agent
from a2c.a2c import ActorCritic, compute_returns

import torch
from torch import optim
import math
import numpy as np
import os
import pickle
import random

n_mues = 1  # number of mues
n_d2d = 2  # number of d2d pairs
n_rb = n_mues  # number of RBs
bs_radius = 500  # bs radius in m

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
STEPS_PER_EPISODE = 5
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.8375*1e-4    # long training
# EPSILON_DECAY = 3.35*1e-4    # medium training
MAX_NUM_EPISODES = 2000      # long training
# MAX_NUM_EPISODES = 480      # medium training
# MAX_NUM_EPISODES = 2      # quick test
ALPHA = 0.2  # Learning rate
GAMMA = 0.98  # Discount factor
# C = 8000 # C constant for the improved reward function
C = 80  # C constant for the improved reward function
TARGET_UPDATE = 10
MAX_NUMBER_OF_AGENTS = 10

HIDDEN_SIZE = 256
LEARNING_RATE = 3e-4
# mu = 0.82*p_max/5/2000
# std = mu/6
mu = p_max*1e-8
std = mu/100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# more parameters
env_params = EnvironmentParameters(
    rb_bandwidth, d2d_pair_distance, p_max, noise_power,
    bs_gain, user_gain, sinr_threshold_train,
    n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin)
train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
reward_function = rewards.dis_reward_tensor
environment = CompleteEnvironmentA2C(env_params, reward_function)
a2c = ActorCritic(6, 1, HIDDEN_SIZE, mu, std)
optimizer = optim.Adam(a2c.parameters(), lr=LEARNING_RATE)

episode = 0
mue_spectral_eff_bag = list()
d2d_spectral_eff_bag = list()
while episode < MAX_NUM_EPISODES:
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0
    best_reward = float('-inf')
    device = torch.device('cuda')
    aux_range = range(MAX_NUMBER_OF_AGENTS+1)[1:]
    n_agents = random.choice(aux_range)
    agents = [Agent() for _ in range(n_agents)]
    environment.build_scenario(agents)
    obs = [environment.get_state(a) for a in agents]
    i = 0
    done = False
    while not done and i < STEPS_PER_EPISODE:
        actions = torch.zeros([len(agents)], device=device)

        for j, agent in enumerate(agents):
            action, dist, value = agent.act(a2c, obs[j])
            actions[j] = action
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)

        next_obs_t, rewards_t, done = environment.step(agents)
        rewards.append(torch.FloatTensor(rewards_t).unsqueeze(1).to(device))
        aux = (1 - done) * torch.ones(n_agents).to(device)
        masks.append(aux)
        i += 1
        obs = next_obs_t

    next_obs = torch.cat(next_obs_t, 0).to(device)
    next_value = list()
    for j, agent in enumerate(agents):
        _, _, next_value_t = agents[0].act(a2c, next_obs_t[j])
        next_value.append(next_value_t)
    next_value = torch.cat(next_value, 0).to(device)
    returns = compute_returns(next_value, rewards, masks)

    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    episode += 1

    print("Episode#:{} mean reward:{}".format(
        episode, torch.mean(torch.cat(rewards)).item()))

# # training function
# # TODO: colocar agente e d2d_device na mesma classe? fazer propriedade d2d_device no agente?
# def train(framework: ExternalDQNFramework, env: CompleteEnvironment2, params: TrainingParameters, agent_params: DQNAgentParameters, max_d2d: int):
#     best_reward = float('-inf')
#     device = torch.device('cuda')
#     mue_spectral_eff_bag = list()
#     d2d_spectral_eff_bag = list()
#     aux_range = range(max_d2d+1)[1:]
#     epsilon = agent_params.start_epsilon
#     for episode in range(params.max_episodes):
#         # TODO: atualmente redistribuo os usuarios aleatoriamente a cada episodio. Isto é o melhor há se fazer? 
#         # Simular deslocamento dos usuários?
#         actions = [i*0.82*p_max/5/1000 for i in range(5)] # best result
#         n_agents = np.random.choice(aux_range)
#         agents = [ExternalDQNAgent(agent_params, actions) for i in range(n_agents)] # 1 agent per d2d tx
#         counts = np.zeros(len(agents))
#         awaits = list()
#         await_steps = [2,3,4]
#         for a in agents:
#             awaits.append(np.random.choice(await_steps))
#             a.set_action(torch.tensor(0).long().cuda(), a.actions[0])
#             a.set_epsilon(epsilon)

#         env.build_scenario(agents)
#         done = False
#         obs = [env.get_state(a) for a in agents] 
#         total_reward = 0.0
#         i = 0
#         bag = list()
#         while not done:
#             if i >= params.steps_per_episode:
#                 break
#             else:
#                 actions = torch.zeros([len(agents)], device=device)
#                 for j, agent in enumerate(agents):
#                     if counts[j] < awaits[j]:
#                         counts[j] += 1
#                     else:
#                         agent.get_action(framework, obs[j])
#                         actions[j] = agent.action_index
#                         counts[j] = 0
#                         awaits[j] = np.random.choice(await_steps)
#                 next_obs, rewards, done = env.step(agents)
#                 i += 1
#                 for j, agent in enumerate(agents):
#                     framework.replay_memory.push(obs[j], actions[j], next_obs[j], rewards[j])
#                 framework.learn()
#                 obs = next_obs
#                 total_reward += torch.sum(rewards)
#                 bag.append(total_reward.item())   
#                 obs = next_obs
#                 if episode % TARGET_UPDATE == 0:
#                     framework.target_net.load_state_dict(framework.policy_net.state_dict())
#             if total_reward > best_reward:
#                 best_reward = total_reward
#             print("Episode#:{} sum reward:{} best_sum_reward:{} eps:{}".format(episode,
#                                      total_reward, best_reward, agents[0].epsilon))

#             # some statistics
#             mue_spectral_eff_bag.append(env.mue_spectral_eff)     # mue spectral eff
#             d2d_spectral_eff_bag.append(env.d2d_spectral_eff/env.params.n_d2d)   # average d2d spectral eff
#         epsilon = agents[0].epsilon

    
#     # Return the trained policy
#     return mue_spectral_eff_bag, d2d_spectral_eff_bag

            
# # SCRIPT EXEC
# # training
# mue_spectral_effs, d2d_spectral_effs = train(ext_framework, environment, train_params, agent_params, MAX_NUMBER_OF_AGENTS)
# spectral_effs = zip(mue_spectral_effs, d2d_spectral_effs)

# cwd = os.getcwd()

# filename = gen.path_leaf(__file__)
# filename = filename.split('.')[0]
# filename_model = filename
# filename = f'{lucas_path}/data/{filename}.pickle'
# torch.save(ext_framework.policy_net.state_dict(), f'{lucas_path}/models/{filename_model}.pt')
# with open(filename, 'wb') as f:
#     pickle.dump(spectral_effs, f)

# # plt.figure(1)
# # plt.plot(mue_spectral_effs, '.', label='MUEs')
# # plt.plot(d2d_spectral_effs, '.', label='D2Ds')
# # plt.xlabel('Iteration')
# # plt.ylabel('Average Spectral Efficiencies')
# # plt.legend()

# plt.figure(1)
# plt.plot(ext_framework.bag, '.')
# plt.xlabel('Iteration')
# plt.ylabel('Average Q-Values')

# plt.show()


