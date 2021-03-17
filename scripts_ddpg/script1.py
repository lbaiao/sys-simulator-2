from sys_simulator.general import print_stuff_ddpg
from sys_simulator.q_learning.environments.completeEnvironment11 import CompleteEnvironment11
from sys_simulator.ddpg.agent import SysSimAgent
from sys_simulator.general.ou_noise import OUNoise, SysSimOUNoise
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from sys_simulator.ddpg.framework import Framework
from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from sys_simulator.devices.devices import db_to_power
from sys_simulator.q_learning.environments.completeEnvironment10dB import CompleteEnvironment10dB
from sys_simulator.channels import BANChannel, UrbanMacroNLOSWinnerChannel
from sys_simulator.parameters.parameters import DQNAgentParameters, EnvironmentParameters, LearningParameters
import torch

# Uses CompleteEnvironment10dB
# Centralized Learning-Centralized Execution
# Trains a central agent for a fixed amount of devices.
# There are different channels to the BS and to the devices.
# Multiple episodes convergence. Everything is in dB.
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
C = 8  # C constant for the improved reward function
ENVIRONMENT_MEMORY = 2
MAX_NUMBER_OF_AGENTS = 3
REWARD_PENALTY = 1.5
# q-learning parameters
# training
REWARD_FUNCTION = 'classic'
MAX_STEPS = 12000
STEPS_PER_EPISODE = 500
REPLAY_INITIAL = int(0E3)
EVAL_NUM_EPISODES = 10
REPLAY_MEMORY_SIZE = int(1E6)
ACTOR_LEARNING_RATE = 1E-4
CRITIC_LEARNING_RATE = 1E-3
HIDDEN_SIZE_1 = 256
HIDDEN_SIZE_2 = 256
BATCH_SIZE = 128
GAMMA = .99
SOFT_TAU = 1E-2
ALPHA = .6
BETA = .4
EXPLORATION = 'ou'
REPLAY_MEMORY_TYPE = 'standard'
PRIO_BETA_ITS = int(.8*(MAX_STEPS - REPLAY_INITIAL))
EVAL_EVERY = int(MAX_STEPS / 20)
OU_DECAY_PERIOD = 100000
OU_MU = 0.0
OU_THETA = .15
OU_MAX_SIGMA = .3
OU_MIN_SIGMA = .3

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
# foo env and foo agents stuff
agent_params = DQNAgentParameters(
    .05, .05, 1, REPLAY_MEMORY_SIZE,
    BATCH_SIZE, GAMMA
)
foo_env = deepcopy(ref_env)
foo_agents = [ExternalDQNAgent(agent_params, [1]) for _ in range(4)]
foo_env.build_scenario(foo_agents)
_, _ = foo_env.step(foo_agents)
a_min = 0
a_max = db_to_power(p_max)
action_size = MAX_NUMBER_OF_AGENTS
env_state_size = MAX_NUMBER_OF_AGENTS * foo_env.get_state_size(foo_agents[0])

framework = Framework(
    REPLAY_MEMORY_TYPE,
    REPLAY_MEMORY_SIZE,
    REPLAY_INITIAL,
    env_state_size,
    action_size,
    HIDDEN_SIZE_1,
    HIDDEN_SIZE_2,
    ACTOR_LEARNING_RATE,
    CRITIC_LEARNING_RATE,
    BATCH_SIZE,
    GAMMA,
    SOFT_TAU,
    torch_device,
    alpha=ALPHA,
    beta=BETA,
    beta_its=PRIO_BETA_ITS
)

ou_noise = SysSimOUNoise(
    action_size,
    a_min, a_max,
    OU_MU, OU_THETA,
    OU_MAX_SIGMA,
    OU_MIN_SIGMA,
    OU_DECAY_PERIOD
)

agent = SysSimAgent(a_min, a_max, EXPLORATION, torch_device)


def train(start: int, writer: SummaryWriter, timestamp: str):
    mue_spectral_eff_bag = list()
    d2d_spectral_eff_bag = list()
    rewards_bag = list()
    step = 0
    while step < MAX_STEPS:
        obs = env.reset()
        env = deepcopy(ref_env)
        now = (time() - start) / 60
        print_stuff_ddpg(step, now, MAX_STEPS, REPLAY_MEMORY_TYPE)
