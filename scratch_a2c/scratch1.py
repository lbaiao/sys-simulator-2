#  Similar to scratch18.py, but the model is trained till 20 devices.
# We also measure how many times the devices chose the same tx power.
# We use script26 model. We simulate for N_D2D=1 till 10

import os

from general import general as gen
from q_learning import rewards
from parameters.parameters import EnvironmentParameters, TrainingParameters
from typing import List
from matplotlib import pyplot as plt

import torch
import numpy as np
import pickle

from q_learning.environments.completeEnvironmentA2C \
    import CompleteEnvironmentA2C
from a2c.agent import Agent
from a2c.a2c import ActorCritic


def test(env: CompleteEnvironmentA2C, framework: ActorCritic,
         max_d2d: int, num_episodes: int, episode_steps: int,
         aux_range: List[int]):
    mue_spectral_effs = [list() for i in range(max_d2d+1)]
    d2d_spectral_effs = [list() for i in range(max_d2d+1)]
    done = False
    bag = list()
    # aux_range = range(max_d2d+1)[1:]
    for _ in range(num_episodes):
        n_agents = np.random.choice(aux_range)
        agents = [Agent() for _ in range(n_agents)]
        env.build_scenario(agents)
        done = False
        obs = [env.get_state(a) for a in agents]
        i = 0
        while not done and i < STEPS_PER_EPISODE:
            for j, agent in enumerate(agents):
                action, dist, value = agent.act(framework, obs[j])
            next_obs, rewards, done = env.step(agents)
            obs = next_obs
            i += 1
        mue_spectral_effs[n_agents].append(env.mue_spectral_eff.item())
        d2d_spectral_effs[n_agents].append(env.d2d_spectral_eff.item())

        # action_counts[n_agents].append(gen.action_counts(env.sinr_d2ds))
    return total_reward, mue_spectral_effs, d2d_spectral_effs, bag


n_mues = 1  # number of mues
n_d2d = 2  # number of d2d pairs
n_rb = n_mues   # number of RBs
bs_radius = 500  # bs radius in m

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
STEPS_PER_EPISODE = 4000
EPSILON_MIN = 0.01
EPSILON_DECAY = 100 * EPSILON_MIN / STEPS_PER_EPISODE
MAX_NUM_EPISODES = int(1.2/EPSILON_DECAY)
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
C = 80  # C constant for the improved reward function
TARGET_UPDATE = 10
MAX_NUMBER_OF_AGENTS = 10
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 128

HIDDEN_SIZE = 256
LEARNING_RATE = 3e-4
# mu = 0.82*p_max/5/2000
# std = mu/6
mu = p_max*1e-8
std = mu/100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# more parameters
cwd = os.getcwd()

env_params = EnvironmentParameters(
    rb_bandwidth, d2d_pair_distance, p_max, noise_power,
    bs_gain, user_gain, sinr_threshold_mue,
    n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin)
train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)

actions = torch.tensor([i*0.82*p_max/5/1000 for i in range(5)])
reward_function = rewards.dis_reward_tensor2
environment = CompleteEnvironmentA2C(env_params, reward_function)

framework = ActorCritic(6, 1, HIDDEN_SIZE, mu, std)
framework.policy_net.load_state_dict(torch.load(f'{cwd}/models/script1.pt'))

reward_function = rewards.dis_reward_tensor

# policy 5 test
aux_range = list(range(MAX_NUMBER_OF_AGENTS+1))[1:]
total_reward, mue_spectral_effs, d2d_spectral_effs, bag, = \
    test(environment, framework, MAX_NUMBER_OF_AGENTS, 2000, 10, aux_range)

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
    log.append('-------------------------------------')

filename = gen.path_leaf(__file__)
filename = filename.split('.')[0]

pickle_filename = f'{cwd}/data/a2c/{filename}.pickle'

data = {
    'd2d_speffs_avg_total': d2d_spectral_effs,
    'mue_success_rate': mue_success_rate,
}

with open(pickle_filename, 'wb') as file:
    pickle.dump(data, file)

filename = f'{cwd}/logs/{filename}.txt'
file = open(filename, 'w')

for lg in log:
    file.write(f'{lg}\n')
file.close()

# fig2, ax1 = plt.subplots()
# ax1.set_xlabel('Number of D2D pairs in the RB')
# ax1.set_ylabel('D2D Average Spectral Efficiency [bps/Hz]', color='tab:blue')
# ax1.plot(d2d_speffs_avg, '.', color='tab:blue')

# ax2 = ax1.twinx()
# ax2.set_ylabel('MUE Success Rate', color='tab:red')
# ax2.plot(mue_success_rate, '.', color='tab:red')
# fig2.tight_layout()

# plt.show()
