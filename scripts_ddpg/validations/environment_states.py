from copy import deepcopy
from sys_simulator.general import power_to_db
import matplotlib.pyplot as plt
from math import ceil, sqrt
import numpy as np
from sys_simulator.q_learning.environments.completeEnvironment11 \
    import CompleteEnvironment11
from sys_simulator.parameters.parameters import EnvironmentParameters
from sys_simulator.channels import BANChannel
from sys_simulator.channels import UrbanMacroNLOSWinnerChannel
from sys_simulator.ddpg.agent import SurrogateAgent
n_mues = 1  # number of mues
n_d2d = 2  # number of d2d pairs
n_rb = n_mues   # number of RBs
carrier_frequency = 2.4  # carrier frequency in GHz
bs_radius = 500  # bs radius in m
rb_bandwidth = 180*1e3  # rb bandwidth in Hz
d2d_pair_distance = 50  # d2d pair distance in m
device_height = 1.5  # mobile devices height in m
bs_height = 25  # BS antenna height in m
p_max = 40  # max tx power in dBm
noise_power = -116  # noise power per RB in dBm
bs_gain = 17    # macro bs antenna gain in dBi
user_gain = 4   # user antenna gain in dBi
sinr_threshold_train = 6  # mue sinr threshold in dB for training
mue_margin = 200  # mue margin in dB
# conversions from dBm to dB
p_max = p_max - 30
noise_power = noise_power - 30
CHANNEL_RND = True
C = 8  # C constant for the improved reward function
ENVIRONMENT_MEMORY = 2
MAX_NUMBER_OF_AGENTS = 3
REWARD_PENALTY = 1.5
REWARD_FUNCTION = 'classic'
MAX_NUM_EPISODES = int(1e4)
STEPS_PER_EPISODE = 5
NUM_BINS = int(1e2)
# ========== END OF PARAMETERS =============
channel_to_bs = UrbanMacroNLOSWinnerChannel(
    rnd=CHANNEL_RND, f_c=carrier_frequency, h_bs=bs_height, h_ms=device_height
)
env_params = EnvironmentParameters(
    rb_bandwidth, d2d_pair_distance, p_max, noise_power,
    bs_gain, user_gain, sinr_threshold_train,
    n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin
)
channel_to_devices = BANChannel(rnd=CHANNEL_RND)
channel_to_bs = UrbanMacroNLOSWinnerChannel(
    rnd=CHANNEL_RND, f_c=carrier_frequency, h_bs=bs_height, h_ms=device_height
)
ref_env = CompleteEnvironment11(
    env_params,
    channel_to_bs,
    channel_to_devices,
    reward_penalty=REWARD_PENALTY,
    memory=ENVIRONMENT_MEMORY,
    bs_height=bs_height,
    reward_function=REWARD_FUNCTION
)
surr_agents = [SurrogateAgent() for _ in range(MAX_NUMBER_OF_AGENTS)]
collected_states = []
rewards = []
mue_speffs = []
d2d_speffs = []
max_speffs = []
env = deepcopy(ref_env)
for episode in range(MAX_NUM_EPISODES):
    env = deepcopy(ref_env)
    env.build_scenario(surr_agents)
    for step in range(STEPS_PER_EPISODE):
        print(f'episode: {episode}; step: {step}')
        actions = np.random.random(MAX_NUMBER_OF_AGENTS)
        actions = power_to_db(actions)
        s_max_speffs = []
        for ag, ac in zip(surr_agents, actions):
            ag.set_action(ac)
            s_max_speffs.append(ag.d2d_tx.max_speffs[0])
        max_speffs += s_max_speffs
        obs_aux, s_rewards, _, _ = env.step(surr_agents)
        rewards += s_rewards
        collected_states += obs_aux
        mue_speffs.append(env.mue_spectral_eff)
        d2d_speffs.append(env.d2d_spectral_eff)
# marcela
# plot distributions
collected_states = np.array(collected_states).reshape(-1, env.state_size())
total_plots = collected_states.shape[1]
n_rows = ceil(sqrt(total_plots))
n_cols = ceil(sqrt(total_plots))
fig, axs = plt.subplots(n_rows, n_cols, sharey=True, tight_layout=True,
                       figsize=(10,10))
index = 0
for i in axs:
    for j in i:
        if index < total_plots-1:
            j.hist(collected_states[:, index], density=True, bins=NUM_BINS)
            index += 1
        else:
            break
    if index >= total_plots-1:
        break
plt.savefig('figs/env_states.png')
# plot rewards distribution
fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True,
                       figsize=(10,10))
axs[0][0].hist(rewards, density=True, bins=NUM_BINS)
axs[0][1].hist(mue_speffs, density=True, bins=NUM_BINS)
axs[1][0].hist(d2d_speffs, density=True, bins=NUM_BINS)
axs[1][1].hist(max_speffs, density=True, bins=NUM_BINS)
plt.savefig('figs/env_rewards.png')
# end
print('done')
