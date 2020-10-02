# Same as script 31, but everything is in dB.
from sys_simulator.general.general import db_to_power, power_to_db
from sys_simulator.channels import BANChannel
from sys_simulator.general import general as gen
from sys_simulator.q_learning.environments.completeEnvironment5dB \
    import CompleteEnvironment5dB
from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from sys_simulator.dqn.externalDQNFramework import ExternalDQNFramework
from sys_simulator.parameters.parameters import \
    EnvironmentParameters, TrainingParameters, DQNAgentParameters
from sys_simulator.q_learning.rewards import dis_reward_tensor_db
import torch
import numpy as np
import os
import pickle


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
sinr_threshold_train = 6  # mue sinr threshold in dB for training
mue_margin = 200  # mue margin in dB
# conversions from dBm to dB
p_max = p_max - 30
noise_power = noise_power - 30
# channel parameters
CHANNEL_RND = False
# q-learning parameters
# training
STEPS_PER_EPISODE = 10
# MAX_NUM_EPISODES = 110      # fast training
MAX_NUM_EPISODES = 550 * 5      # fast training
# MAX_NUM_EPISODES = 10      # debugging
# testing
# TEST_NUM_EPISODES = 110
TEST_NUM_EPISODES = 550 * 5
# TEST_NUM_EPISODES = 1  # testing
TEST_STEPS_PER_EPISODE = 10
# common
EPSILON_MIN = 0.05
# EPSILON_DECAY = 1e-3    # fast training
EPSILON_DECAY = .2e-3 / 4    # fast training
# EPSILON_DECAY = .4167*1e-3    # long training
# EPSILON_DECAY = .04167*1e-3    # super long training
# GAMMA = 0.98  # Discount factor
GAMMA = 0.9  # Discount factor
C = 8  # C constant for the improved reward function
TARGET_UPDATE = 10
MAX_NUMBER_OF_AGENTS = 10
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 256
HIDDEN_SIZE = 64
NUM_HIDDEN_LAYERS = 5
max_d2d = MAX_NUMBER_OF_AGENTS
# more parameters
# linear discretization
# actions = power_to_db(np.linspace(
#     db_to_power(p_max-20), db_to_power(p_max-10), 10
# ))
# db discretization
actions = np.linspace(p_max-28, p_max-18, 10)
actions[0] = -1000
env_params = EnvironmentParameters(
    rb_bandwidth, d2d_pair_distance, p_max, noise_power,
    bs_gain, user_gain, sinr_threshold_train,
    n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin
)
params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
agent_params = DQNAgentParameters(
    EPSILON_MIN, EPSILON_DECAY, 1, REPLAY_MEMORY_SIZE,
    BATCH_SIZE, GAMMA
)
reward_function = dis_reward_tensor_db
channel = BANChannel(rnd=CHANNEL_RND)
env = CompleteEnvironment5dB(env_params, reward_function, channel)
foo_agent = ExternalDQNAgent(agent_params, [1])
env.build_scenario([foo_agent])
env_state_size = env.get_state_size(foo_agent)
framework = ExternalDQNFramework(
    agent_params,
    env_state_size,
    len(actions),
    HIDDEN_SIZE,
    NUM_HIDDEN_LAYERS
)


def train():
    global actions
    best_reward = float('-inf')
    device = torch.device('cuda')
    mue_spectral_eff_bag = list()
    d2d_spectral_eff_bag = list()
    aux_range = range(max_d2d+1)[1:]
    epsilon = agent_params.start_epsilon
    for episode in range(params.max_episodes):
        n_agents = np.random.choice(aux_range)
        agents = [ExternalDQNAgent(agent_params, actions)
                  for _ in range(n_agents)]  # 1 agent per d2d tx
        for a in agents:
            a.set_epsilon(epsilon)
        env.build_scenario(agents)
        obs = [env.get_state(a).float() for a in agents]
        total_reward = 0.0
        i = 0
        bag = list()
        while True:
            if i >= params.steps_per_episode:
                break
            else:
                past_actions = torch.zeros([len(agents)], device=device)
                for j, agent in enumerate(agents):
                    agent.get_action(framework, obs[j].float())
                    past_actions[j] = agent.action_index
                # # debugging
                # if len(agents) == 2:
                #     print('debugging')
                next_obs, rewards = env.step(agents)
                i += 1
                for j, agent in enumerate(agents):
                    framework.replay_memory.push(
                        obs[j].float(), past_actions[j],
                        next_obs[j].float(), rewards[j]
                    )
                framework.learn()
                obs = next_obs
                total_reward += np.sum(rewards)
                bag.append(total_reward.item())
                obs = next_obs
                if episode % TARGET_UPDATE == 0:
                    framework.target_net.load_state_dict(
                        framework.policy_net.state_dict()
                    )
            if total_reward > best_reward:
                best_reward = total_reward
            print("Episode#:{} sum reward:{} best_sum_reward:{} eps:{}".format(
                episode, total_reward, best_reward, agents[0].epsilon)
            )
            # mue spectral eff
            mue_spectral_eff_bag.append(
                (env.mue_spectral_eff, n_agents)
            )
            # average d2d spectral eff
            d2d_spectral_eff_bag.append(
                (env.d2d_spectral_eff/n_agents, n_agents)
            )
        epsilon = agents[0].epsilon
    # Return the trained policy
    mue_spectral_effs = mue_spectral_eff_bag
    d2d_spectral_effs = d2d_spectral_eff_bag
    spectral_effs = zip(mue_spectral_effs, d2d_spectral_effs)
    avg_q_values = framework.bag
    # saving the data and the model
    cwd = os.getcwd()
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    filename_model = filename
    filename = f'{cwd}/data/dql/{filename}_training.pt'
    torch.save(framework.policy_net.state_dict(),
               f'{cwd}/models/dql/{filename_model}.pt')
    torch.save(spectral_effs, filename)
    with open(
        f'{cwd}/data/dql/{filename_model}_avg_q_values.pickle',
        'wb'
    ) as p_file:
        pickle.dump(avg_q_values, p_file)


def test():
    framework.policy_net.load_state_dict(
        torch.load('models/dql/script32.pt')
    )
    mue_spectral_effs = [list() for _ in range(max_d2d+1)]
    d2d_spectral_effs = [list() for _ in range(max_d2d+1)]
    # jain_index = [list() for _ in range(max_d2d+1)]
    done = False
    bag = list()
    aux_range = range(max_d2d+1)[1:]
    for _ in range(TEST_NUM_EPISODES):
        n_agents = np.random.choice(aux_range)
        agents = [ExternalDQNAgent(agent_params, actions)
                  for i in range(n_agents)]  # 1 agent per d2d tx
        env.build_scenario(agents)
        done = False
        obs = [env.get_state(a) for a in agents]
        total_reward = 0.0
        i = 0
        while not done:
            for j, agent in enumerate(agents):
                aux = agent.act(framework, obs[j].float()).max(1)
                agent.set_action(aux[1].long(),
                                 agent.actions[aux[1].item()])
                bag.append(aux[1].item())
            next_obs, rewards = env.step(agents)
            obs = next_obs
            total_reward += sum(rewards)
            i += 1
            if i >= TEST_STEPS_PER_EPISODE:
                break
        mue_spectral_effs[n_agents].append(env.mue_spectral_eff.item())
        d2d_spectral_effs[n_agents].append(env.d2d_spectral_eff.item())
        # jain_index[n_agents].append(gen.jain_index(env.sinr_d2ds))
    mue_success_rate = list()
    for i, m in enumerate(mue_spectral_effs):
        mue_success_rate.append(
            np.average(m > np.log2(1 + sinr_threshold_train))
        )
    d2d_speffs_avg = list()
    for i, d in enumerate(d2d_spectral_effs):
        d2d_speffs_avg.append(np.average(d))
    # jain_index_avg = list()
    # for i, j in enumerate(jain_index):
    #     jain_index_avg.append(np.average(j))
    log = list()
    for i, d in enumerate(zip(d2d_speffs_avg, mue_success_rate)):
        log.append(f'NUMBER OF D2D_USERS: {i+1}')
        log.append(f'D2D SPECTRAL EFFICIENCY - SCRIPT: {d[0]}')
        log.append(f'MUE SUCCESS RATE - SCRIPT: {d[1]}')
        log.append('-------------------------------------')
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    log_path = f'logs/dql/{filename}.txt'
    file = open(log_path, 'w')
    for lg in log:
        file.write(f'{lg}\n')
    file.close()
    data_path = f'data/dql/{filename}.pickle'
    data = {
        'd2d_speffs_avg_total': d2d_spectral_effs,
        'mue_success_rate': mue_success_rate,
        'chosen_actions': bag,
    }
    with open(data_path, 'wb') as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    train()
    test()
