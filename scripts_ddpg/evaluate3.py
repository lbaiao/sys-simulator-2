from copy import deepcopy
from shutil import copyfile
from typing import List
from time import time
from sys_simulator.parameters.parameters import EnvironmentParameters
from sys_simulator.channels import BANChannel, UrbanMacroNLOSWinnerChannel
from sys_simulator.ddpg.agent import SurrogateAgent, SysSimAgent
from sys_simulator.q_learning.environments.completeEnvironment12 import CompleteEnvironment12
import sys_simulator.general as gen
from sys_simulator.general import load_with_pickle, print_evaluate3, random_seed, save_with_pickle
from sys_simulator.ddpg.framework import Framework
import torch


# parameters
ALGO_NAME = 'ddpg'
BASE_PATH = '/home/lucas/dev/sys-simulator-2'
AGENTS_RANGE = range(6)[1:]
MODELS_PATHS = [
    f'{BASE_PATH}/data/ddpg/script7/20210513-203713/last_model.pt',
    f'{BASE_PATH}/data/ddpg/script7/20210513-212829/last_model.pt',
    f'{BASE_PATH}/data/ddpg/script7/20210511-204603/last_model.pt',
    f'{BASE_PATH}/data/ddpg/script7/20210513-233135/last_model.pt',
    f'{BASE_PATH}/data/ddpg/script7/20210511-223017/last_model.pt',
]
# ENVS_PATHS = [
    # f'{BASE_PATH}/data/ddpg/script7/20210509-171944/env.pickle',
    # f'{BASE_PATH}/data/ddpg/script7/20210506-100648/env.pickle',
    # f'{BASE_PATH}/data/ddpg/script7/20210509-182131/env.pickle',
    # f'{BASE_PATH}/data/ddpg/script7/20210509-190508/env.pickle',
    # f'{BASE_PATH}/data/ddpg/script7/20210509-210122/env.pickle',
# ]
# TEST_NUM_EPISODES = 10000
TEST_NUM_EPISODES = 1000
EVAL_STEPS_PER_EPISODE = 10
PRINT_EVERY = 100
# env parameters
RND_SEED = True
SEED = 42
CHANNEL_RND = True
# writer
filename = gen.path_leaf(__file__)
filename = filename.split('.')[0]
dir_path = f'data/{ALGO_NAME}/{filename}'
data_path, _ = gen.make_dir_timestamp(dir_path)
if RND_SEED:
    random_seed(SEED)
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a_min = -90
a_max = 60
a_offset = -10
frameworks = []
for p in MODELS_PATHS:
    f = torch.load(p, map_location=torch_device)
    frameworks.append(f)
# envs = [load_with_pickle(p) for p in ENVS_PATHS]
central_agent_test = SysSimAgent(a_min, a_max, 'perturberd',
                                 torch_device, a_offset=a_offset)
# env parameters
n_mues = 1  # number of mues
n_rb = n_mues   # number of RBs
carrier_frequency = 2.4  # carrier frequency in GHz
bs_radius = 1000  # bs radius in m
rb_bandwidth = 180*1e3  # rb bandwidth in Hz
d2d_pair_distance = 5  # d2d pair distance in m
device_height = 1.5  # mobile devices height in m
bs_height = 25  # BS antenna height in m
p_max = 40  # max tx power in dBm
noise_power = -116  # noise power per RB in dBm
bs_gain = 17    # macro bs antenna gain in dBi
user_gain = 4   # user antenna gain in dBi
sinr_threshold_train = 6  # mue sinr threshold in dB for training
mue_margin = 6
MIN_D2D_PAIR_DISTANCE = 1.5
MAX_D2D_PAIR_DISTANCE = 15
# conversions from dBm to dB
p_max = p_max - 30
noise_power = noise_power - 30
# env parameters
RND_SEED = True
SEED = 42
CHANNEL_RND = True
C = 8  # C constant for the improved reward function
ENVIRONMENT_MEMORY = 2
MAX_NUMBER_OF_AGENTS = 2
REWARD_PENALTY = 1.5
DELTA_T = .5
# q-learning parameters
# training
REWARD_FUNCTION = 'jain'
STATES_OPTIONS = ['sinrs', 'positions', 'channels']
env_params = EnvironmentParameters(
    rb_bandwidth, None, p_max, noise_power,
    bs_gain, user_gain, sinr_threshold_train,
    n_mues, MAX_NUMBER_OF_AGENTS, n_rb, bs_radius,
    c_param=C, mue_margin=mue_margin,
    min_d2d_pair_distance=MIN_D2D_PAIR_DISTANCE,
    max_d2d_pair_distance=MAX_D2D_PAIR_DISTANCE
)
channel_to_devices = BANChannel(rnd=CHANNEL_RND)
channel_to_bs = UrbanMacroNLOSWinnerChannel(
    rnd=CHANNEL_RND, f_c=carrier_frequency, h_bs=bs_height, h_ms=device_height,
    small_sigma=4.0, sigma=8.0
)
ref_env = CompleteEnvironment12(
    env_params,
    channel_to_bs,
    channel_to_devices,
    reward_penalty=REWARD_PENALTY,
    memory=ENVIRONMENT_MEMORY,
    bs_height=bs_height,
    reward_function=REWARD_FUNCTION,
    states_options=STATES_OPTIONS,
    memories_capacity=int(1e3),
    dt=DELTA_T
)


def test(framework: Framework,
         surr_agents: List[SurrogateAgent], start: float):
    framework.actor.eval()
    framework.critic.eval()
    mue_availability = []
    mue_sinrs = []
    d2d_sinrs = []
    rewards_bag = []
    for ep in range(TEST_NUM_EPISODES):
        if ep % PRINT_EVERY == 0:
            now = (time() - start) / 60
            print_evaluate3(ep, TEST_NUM_EPISODES, now, len(surr_agents))
        env = deepcopy(ref_env)
        env.build_scenario(surr_agents, motion_model='random')
        obs, _, _, _ = env.step(surr_agents)
        i = 0
        done = False
        ep_availability = []
        ep_rewards = []
        ep_mue_sinrs = []
        ep_d2d_sinrs = []
        while not done and i < EVAL_STEPS_PER_EPISODE:
            actions = central_agent_test.act(obs, framework, False)
            # actions = np.zeros(MAX_NUMBER_OF_AGENTS) + 1e-9
            # db_actions = power_to_db(actions)
            db_actions = actions
            for j, agent in enumerate(surr_agents):
                agent.set_action(db_actions[j].item())
            next_obs, reward, done, _ = env.step(surr_agents)
            obs = next_obs
            ep_availability.append(env.mue.sinr > env.params.sinr_threshold)
            ep_rewards.append(reward)
            ep_mue_sinrs.append(env.mue.sinr)
            ep_d2d_sinrs.append([p[0].sinr for p in env.d2d_pairs])
            i += 1
        rewards_bag += ep_rewards
        mue_sinrs += ep_mue_sinrs
        d2d_sinrs += ep_d2d_sinrs
        mue_availability += ep_availability
    all_bags = {
        'rewards': rewards_bag,
        'mue_sinrs': mue_sinrs,
        'd2d_sinrs': d2d_sinrs,
        'mue_availability': mue_availability
    }
    return all_bags


if __name__ == '__main__':
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    dir_path = f'data/{ALGO_NAME}/{filename}'
    data_path, _ = gen.make_dir_timestamp(dir_path)
    start = time()
    results = []
    for f, i in zip(frameworks, AGENTS_RANGE):
        surr_agents = [SurrogateAgent() for _ in range(i)]
        r = test(f, surr_agents, start)
        results.append(r)
    # save stuff
    now = (time() - start) / 60
    data_file_path = f'{data_path}/log.pickle'
    save_with_pickle(results, data_file_path)
    copyfile(__file__, f'{data_path}/{filename}.py')
    print(f'done. Elapsed time: {now} minutes.')

