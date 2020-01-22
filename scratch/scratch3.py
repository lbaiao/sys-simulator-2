import sys

import os
lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from q_learning.environments.distributedEnvironment import DistributedEnvironment
from q_learning.environments.actionEnvironment import ActionEnvironment
from q_learning.environments.environment import RLEnvironment
from q_learning.environments.distanceEnvironment import DistanceEnvironment
from q_learning.agents.agent import Agent
from q_learning.agents.distanceAgent import DistanceAgent
from q_learning.rewards import dis_reward, centralized_reward
from parameters.parameters import EnvironmentParameters, TrainingParameters, AgentParameters, LearningParameters
from general import general as gen
from plots.plots import plot_spectral_effs
from typing import List
from matplotlib import pyplot as plt

import numpy as np

def test1(agents: List[Agent], env: RLEnvironment, policy: np.array, num_episodes: int, episode_steps: int):    
    mue_spectral_effs = list()
    d2d_spectral_effs = list()
    for _ in range(num_episodes):        
        i = 0    
        env.reset(agents)
        done = False
        obs = env.get_state()
        total_reward = 0.0
        while not done:        
            action_index = policy[obs]
            for agent in agents:
                agent.set_action(action_index)
            next_obs, reward, done = env.step(agents)
            obs = next_obs
            total_reward += reward
            i +=1
            if i >= episode_steps:
                break
        mue_spectral_effs.append(env.mue_spectral_eff)
        d2d_spectral_effs.append(env.d2d_spectral_eff)
    return total_reward, mue_spectral_effs, d2d_spectral_effs

def test2(agents: List[Agent], env: DistributedEnvironment, policies, num_episodes: int, episode_steps: int):
    mue_spectral_effs = list()
    d2d_spectral_effs = list()    
    for _ in range(num_episodes):
        env.reset(agents)
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
            if i >= episode_steps:
                break
        mue_spectral_effs.append(env.mue_spectral_eff)
        d2d_spectral_effs.append(env.d2d_spectral_eff)
    return total_reward, mue_spectral_effs, d2d_spectral_effs

def test5(agents: List[Agent], env: ActionEnvironment, policies: np.array, num_episodes: int, episode_steps: int):
    mue_spectral_effs = list()
    d2d_spectral_effs = list()    
    done = False
    for _ in range(num_episodes):
        env.reset(agents)
        done = False
        obs = [env.get_state(a) for a in agents] 
        total_reward = 0.0
        i = 0
        while not done:            
            actions_indexes = np.zeros(len(agents), dtype=int)        
            for m in range(len(agents)):
                actions_indexes[m] = policies[m][obs[m]]
                agents[m].set_action(actions_indexes[m])
            next_obs, rewards, done = env.step(agents)
            obs = next_obs
            total_reward += sum(rewards)
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
                                        n_mues, n_d2d, n_rb, bs_radius, c_param=C)
train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
agent_params = AgentParameters(EPSILON_MIN, EPSILON_DECAY, 1)
learn_params = LearningParameters(ALPHA, GAMMA)

actions = [i*p_max/10 + 1e-9 for i in range(11)]
agents = [Agent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
reward_function1 = centralized_reward
reward_function2 = dis_reward
reward_function5 = dis_reward

environment1 = RLEnvironment(env_params, reward_function1, done_disable=True)
environment2 = DistributedEnvironment(env_params, reward_function2, done_disable=True)
environment5 = ActionEnvironment(env_params, reward_function5, done_disable=True)
environment6 = DistanceEnvironment(env_params, reward_function5, done_disable=True)

learned_policies_1 = np.load(f'D:/Dev/sys-simulator-2/models/model1.npy')
learned_policies_2 = np.load(f'D:/Dev/sys-simulator-2/models/model2.npy')
learned_policies_5 = np.load(f'D:/Dev/sys-simulator-2/models/model5.npy')
learned_policies_6 = np.load(f'D:/Dev/sys-simulator-2/models/model6.npy')


# policy 1 test
t_agents = [Agent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
total_reward, mue_spectral_effs1, d2d_spectral_effs1 = test1(t_agents, environment1, learned_policies_1, 1000, 5)
    
# policy 2 test
t_agents = [Agent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
total_reward, mue_spectral_effs2, d2d_spectral_effs2 = test2(t_agents, environment2, learned_policies_2, 1000, 5)

# policy 5 test
t_agents = [Agent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
total_reward, mue_spectral_effs5, d2d_spectral_effs5 = test5(t_agents, environment5, learned_policies_5, 1000, 5)

t_agents = [DistanceAgent(agent_params, actions) for i in range(n_d2d)] # 1 agent per d2d tx
total_reward, mue_spectral_effs6, d2d_spectral_effs6 = test5(t_agents, environment6, learned_policies_6, 1000, 5)


mue_spectral_effs1 = np.array(mue_spectral_effs1)
mue_spectral_effs1 = np.reshape(mue_spectral_effs1, np.prod(mue_spectral_effs1.shape))
mue_spectral_effs2 = np.array(mue_spectral_effs2)
mue_spectral_effs2 = np.reshape(mue_spectral_effs2, np.prod(mue_spectral_effs2.shape))
mue_spectral_effs5 = np.array(mue_spectral_effs5)
mue_spectral_effs5 = np.reshape(mue_spectral_effs5, np.prod(mue_spectral_effs5.shape))
mue_spectral_effs6 = np.array(mue_spectral_effs6)
mue_spectral_effs6 = np.reshape(mue_spectral_effs6, np.prod(mue_spectral_effs6.shape))

d2d_spectral_effs1 = np.array(d2d_spectral_effs1)
d2d_spectral_effs1 = np.reshape(d2d_spectral_effs1, np.prod(d2d_spectral_effs1.shape))
d2d_spectral_effs2 = np.array(d2d_spectral_effs2)
d2d_spectral_effs2 = np.reshape(d2d_spectral_effs2, np.prod(d2d_spectral_effs2.shape))
d2d_spectral_effs5 = np.array(d2d_spectral_effs5)
d2d_spectral_effs5 = np.reshape(d2d_spectral_effs5, np.prod(d2d_spectral_effs5.shape))
d2d_spectral_effs6 = np.array(d2d_spectral_effs6)
d2d_spectral_effs6 = np.reshape(d2d_spectral_effs6, np.prod(d2d_spectral_effs6.shape))


plt.figure(1)
plt.plot(list(range(len(d2d_spectral_effs1))), d2d_spectral_effs1, '.', label='Script 1')
plt.plot(list(range(len(d2d_spectral_effs2))), d2d_spectral_effs2, '.', label='Script 2')
plt.plot(list(range(len(d2d_spectral_effs5))), d2d_spectral_effs5, '.', label='Script 5')
plt.plot(list(range(len(d2d_spectral_effs6))), d2d_spectral_effs6, '.', label='Script 6')
plt.title('D2D spectral efficiencies')
plt.legend()

plt.figure(2)
threshold_eff = np.log2(1 + sinr_threshold) * np.ones(len(mue_spectral_effs5))
plt.plot(list(range(len(mue_spectral_effs1))), mue_spectral_effs1, '.', label='Script 1')
plt.plot(list(range(len(mue_spectral_effs2))), mue_spectral_effs2, '.', label='Script 2')
plt.plot(list(range(len(mue_spectral_effs5))), mue_spectral_effs5, '.', label='Script 5')
plt.plot(list(range(len(mue_spectral_effs6))), mue_spectral_effs6, '.', label='Script 6')
plt.plot(list(range(len(mue_spectral_effs5))), threshold_eff, label='Threshold')    

plt.title('MUE spectral efficiencies')
plt.legend()
plt.show()

