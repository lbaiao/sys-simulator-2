from shutil import copyfile
import sys_simulator.general as gen
from time import time
from sys_simulator.general import print_stuff_ddpg
from sys_simulator.q_learning.environments.completeEnvironment11 \
    import CompleteEnvironment11
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

# Uses CompleteEnvironment11
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
C = 8  # C constant for the improved reward function
ENVIRONMENT_MEMORY = 2
MAX_NUMBER_OF_AGENTS = 3
REWARD_PENALTY = 1.5
# q-learning parameters
# training
ALGO_NAME = 'ddpg'
REWARD_FUNCTION = 'classic'
MAX_STEPS = 12000
STEPS_PER_EPISODE = 100
# STEPS_PER_EPISODE = MAX_STEPS
REPLAY_INITIAL = int(1E3)
EVAL_NUM_EPISODES = 10
REPLAY_MEMORY_SIZE = int(1E4)
ACTOR_LEARNING_RATE = 5E-5
CRITIC_LEARNING_RATE = 5E-4
HIDDEN_SIZE = 256
N_HIDDEN_LAYERS = 5
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
a_min = 0 + 1e-9
a_max = db_to_power(p_max - 10)
action_size = MAX_NUMBER_OF_AGENTS
env_state_size = MAX_NUMBER_OF_AGENTS * ref_env.state_size()

framework = Framework(
    REPLAY_MEMORY_TYPE,
    REPLAY_MEMORY_SIZE,
    REPLAY_INITIAL,
    env_state_size,
    action_size,
    HIDDEN_SIZE,
    N_HIDDEN_LAYERS,
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

central_agent = SysSimAgentWriter(a_min, a_max, EXPLORATION, torch_device)
central_agent_test = SysSimAgent(a_min, a_max, EXPLORATION, torch_device)
surr_agents = [SurrogateAgent() for _ in range(MAX_NUMBER_OF_AGENTS)]


def train(start: int, writer: SummaryWriter, timestamp: str):
    actor_losses_bag = list()
    critic_losses_bag = list()
    mue_spectral_eff_bag = list()
    d2d_spectral_eff_bag = list()
    rewards_bag = list()
    mue_avail_bag = list()
    step = 0
    while step < MAX_STEPS:
        env = deepcopy(ref_env)
        env.build_scenario(surr_agents)
        obs_aux, _, _, _ = env.step(surr_agents)
        obs = np.concatenate(obs_aux, axis=1)
        now = (time() - start) / 60
        print_stuff_ddpg(step, now, MAX_STEPS, REPLAY_MEMORY_TYPE)
        total_reward = 0.0
        done = False
        i = 0
        while not done and i < STEPS_PER_EPISODE:
            actions = central_agent.act(obs, framework,
                                        writer, step, True, step=i, ou=ou_noise)
            for j, agent in enumerate(surr_agents):
                agent.set_action(actions[0][j])
            next_obs_aux, rewards, done, _ = env.step(surr_agents)
            total_reward = np.sum(rewards)
            next_obs = np.concatenate(next_obs_aux, axis=1)
            framework.replay_memory.push(obs, actions, total_reward, next_obs,
                                         done)
            actor_loss, critic_loss = framework.learn()
            obs = next_obs
            i += 1
            step += 1
            writer.add_scalar('Actor Losses', actor_loss, step)
            writer.add_scalar('Critic Losses', critic_loss, step)
            aux_actions = actions.cpu().numpy()
            writer.add_scalar('Average Actions', aux_actions.mean(), step)
            # writer.add_scalar('Aggregated Actions', aux_actions.sum(), step)
            actor_losses_bag.append(actor_loss)
            critic_losses_bag.append(critic_loss)
            framework.actor.train()
            framework.critic.train()
        if step % EVAL_EVERY == 0:
            t_bags = test(framework)
            t_rewards = t_bags['rewards']
            t_mue_spectral_effs = t_bags['mue_spectral_effs']
            t_d2d_spectral_effs = t_bags['d2d_spectral_effs']
            t_availability = t_bags['mue_availability']
            # mue spectral eff
            mue_spectral_eff_bag.append(t_mue_spectral_effs)
            # average d2d spectral eff
            d2d_spectral_eff_bag.append(t_d2d_spectral_effs)
            rewards_bag.append(t_rewards)
            mue_avail_bag.append(t_availability)
            # write metrics
            writer.add_scalar('Average MUE Spectral efficiencies',
                              np.mean(t_mue_spectral_effs), step)
            writer.add_scalar('Average D2D Spectral Efficiencies',
                              np.mean(t_d2d_spectral_effs), step)
            writer.add_scalar('Aggregated Rewards', np.sum(t_rewards), step)
            writer.add_scalar('Average MUE Availability',
                              np.mean(t_availability), step)
    all_bags = {
        'actor_losses': actor_losses_bag,
        'critic_losses': critic_losses_bag,
        'mue_spectral_effs': mue_spectral_eff_bag,
        'd2d_spectral_effs': d2d_spectral_eff_bag
    }
    return all_bags


def test(framework: Framework):
    surr_agents = [SurrogateAgent() for _ in range(MAX_NUMBER_OF_AGENTS)]
    framework.actor.eval()
    framework.critic.eval()
    mue_availability = []
    mue_spectral_effs = []
    d2d_spectral_effs = []
    rewards_bag = []
    for _ in range(EVAL_NUM_EPISODES):
        env = deepcopy(ref_env)
        env.build_scenario(surr_agents)
        obs_aux, _, _, _ = env.step(surr_agents)
        obs = np.concatenate(obs_aux, axis=1)
        i = 0
        done = False
        ep_availability = []
        ep_rewards = []
        ep_mue_speffs = []
        ep_d2d_speffs = []
        while not done and i < STEPS_PER_EPISODE:
            actions = central_agent_test.act(obs, framework, False)
            for j, agent in enumerate(surr_agents):
                agent.set_action(actions[0][j])
            next_obs_aux, rewards, done, _ = env.step(surr_agents)
            next_obs = np.concatenate(next_obs_aux, axis=1)
            obs = next_obs
            ep_availability.append(env.mue.sinr > env.params.sinr_threshold)
            ep_rewards.append(np.sum(rewards))
            ep_mue_speffs.append(env.mue_spectral_eff)
            ep_d2d_speffs.append(env.d2d_spectral_eff)
            i += 1
        rewards_bag.append(np.sum(ep_rewards))
        mue_spectral_effs.append(np.mean(ep_mue_speffs))
        d2d_spectral_effs.append(np.mean(ep_d2d_speffs))
        mue_availability.append(np.mean(ep_availability))
    all_bags = {
        'rewards': rewards_bag,
        'mue_spectral_effs': mue_spectral_effs,
        'd2d_spectral_effs': d2d_spectral_effs,
        'mue_availability': mue_availability
    }
    return all_bags


def run():
    # make data dir
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    dir_path = f'data/{ALGO_NAME}/{filename}'
    data_path, timestamp = gen.make_dir_timestamp(dir_path)
    writer = SummaryWriter(f'{data_path}/tensorboard')
    start = time()
    train_bags = train(start, writer, timestamp)
    writer.close()
    test_bags = test(framework)
    # save stuff
    now = (time() - start) / 60
    data_file_path = f'{data_path}/log.pickle'
    data = {
        'train_bags': train_bags,
        'test_bags': test_bags,
        'elapsed_time': now,
        'eval_every': EVAL_EVERY,
    }
    gen.save_with_pickle(data, data_file_path)
    copyfile(__file__, f'{data_path}/{filename}.py')
    print(f'done. Elapsed time: {now} minutes.')


if __name__ == '__main__':
    run()
