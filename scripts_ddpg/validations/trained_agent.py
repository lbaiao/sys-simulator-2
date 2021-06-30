from shutil import copyfile
from sys_simulator.plots import plot_env_states
import sys_simulator.general as gen
from time import time
from sys_simulator.general import power_to_db, print_evaluating, print_stuff_ddpg
from sys_simulator.q_learning.environments.completeEnvironment12 \
    import CompleteEnvironment12
from sys_simulator.ddpg.agent import SysSimAgent, SurrogateAgent, SysSimAgentWriter
from sys_simulator.general.ou_noise import SysSimOUNoise
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from sys_simulator.ddpg.framework import Framework
from sys_simulator.devices.devices import db_to_power
from sys_simulator.channels import BANChannel, UrbanMacroNLOSWinnerChannel
from sys_simulator.parameters.parameters import EnvironmentParameters
import torch
import numpy as np


# Uses CompleteEnvironment12
# Centralized Learning-Centralized Execution
# Trains a central agent for a fixed amount of devices.
# There are different channels to the BS and to the devices.
# Multiple episodes convergence. The states are in linear scale.
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
# env parameters
CHANNEL_RND = True
C = 2  # C constant for the improved reward function
ENVIRONMENT_MEMORY = 2
MAX_NUMBER_OF_AGENTS = 3
REWARD_PENALTY = 1.5
# q-learning parameters
# training
MODEL_PATH = 'models/ddpg/script3/20210327-201621/framework.pt'
ALGO_NAME = 'ddpg'
REWARD_FUNCTION = 'classic'
MAX_STEPS = 1000
STEPS_PER_EPISODE = 50
EXPLORATION = 'ou'
# devices positions
pairs_positions = [
    ((-400, 0, device_height), (-450, 0, device_height)),
    ((225, 225, device_height), (275, 225, device_height)),
    ((55, -55, device_height), (55, -5, device_height)),
]
mue_position = (0, 200, device_height)


torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env_params = EnvironmentParameters(
    rb_bandwidth, d2d_pair_distance, p_max, noise_power,
    bs_gain, user_gain, sinr_threshold_train,
    n_mues, MAX_NUMBER_OF_AGENTS, n_rb, bs_radius, c_param=C, mue_margin=mue_margin
)
channel_to_devices = BANChannel(rnd=CHANNEL_RND)
channel_to_bs = UrbanMacroNLOSWinnerChannel(
    rnd=CHANNEL_RND, f_c=carrier_frequency, h_bs=bs_height, h_ms=device_height
)
ref_env = CompleteEnvironment12(
    env_params,
    channel_to_bs,
    channel_to_devices,
    reward_penalty=REWARD_PENALTY,
    memory=ENVIRONMENT_MEMORY,
    bs_height=bs_height,
    reward_function=REWARD_FUNCTION,
    memories_capacity=10
)
a_min = 0 + 1e-9
a_max = db_to_power(p_max - 10)
action_size = MAX_NUMBER_OF_AGENTS
env_state_size = ref_env.state_size()

framework: Framework = torch.load(MODEL_PATH)
central_agent = SysSimAgentWriter(a_min, a_max, EXPLORATION, torch_device)
central_agent_test = SysSimAgent(a_min, a_max, EXPLORATION, torch_device)
surr_agents = [SurrogateAgent() for _ in range(MAX_NUMBER_OF_AGENTS)]


def evaluate(start: float, writer: SummaryWriter):
    step = 0
    while step < MAX_STEPS:
        env = deepcopy(ref_env)
        # env.set_scenario(pairs_positions, mue_position, surr_agents)
        env.build_scenario(surr_agents)
        obs, _, _, _ = env.step(surr_agents)
        now = (time() - start) / 60
        print_evaluating(step, now, MAX_STEPS)
        done = False
        i = 0
        while not done and i < STEPS_PER_EPISODE:
            actions = central_agent.act(obs, framework,
                                        writer, step, False)
            db_actions = power_to_db(actions.cpu().numpy())
            for j, agent in enumerate(surr_agents):
                agent.set_action(db_actions[j].item())
            next_obs, reward, done, _ = env.step(surr_agents)
            framework.replay_memory.push(obs, actions, reward, next_obs,
                                         done)
            obs = next_obs
            i += 1
            step += 1
            writer.add_scalar('Rewards', reward, step)
            sinrs = [a.d2d_tx.sinr for a in surr_agents]
            sinrs = {f'device {i}': s for i, s in enumerate(sinrs)}
            writer.add_scalars('SINRs', sinrs, step)
            writer.add_scalars(
                'Transmission powers',
                {f'device {i}': a for i, a in enumerate(db_actions)},
                step
            )
            mue_success = int(env.mue.sinr > env.params.sinr_threshold)
            writer.add_scalar('MUE success', mue_success, step)


def run():
    # make data dir
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    dir_path = f'data/{ALGO_NAME}/validations/{filename}'
    data_path, timestamp = gen.make_dir_timestamp(dir_path)
    writer = SummaryWriter(f'{data_path}/tensorboard')
    start = time()
    evaluate(start, writer)
    writer.close()
    # save stuff
    now = (time() - start) / 60
    print(f'done. Elapsed time: {now} minutes.')


if __name__ == '__main__':
    run()
