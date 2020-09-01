#  Testing for script 31
import pickle
from sys_simulator.channels import BANChannel
from sys_simulator.general import general as gen
from sys_simulator.q_learning.environments.completeEnvironment5\
    import CompleteEnvironment5
from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from sys_simulator.dqn.externalDQNFramework import ExternalDQNFramework
from sys_simulator.q_learning import rewards as reward_func
from sys_simulator.parameters.parameters \
    import EnvironmentParameters, DQNAgentParameters
import torch
import numpy as np


def run():
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
    STEPS_PER_EPISODE = 25
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 1
    MAX_NUM_EPISODES = 2000
    # MAX_NUM_EPISODES = 2  # test
    GAMMA = 0.98  # Discount factor
    C = 8  # C constant for the improved reward function
    MAX_NUMBER_OF_AGENTS = 10
    REPLAY_MEMORY_SIZE = 10000
    BATCH_SIZE = 128
    # more parameters
    max_d2d = MAX_NUMBER_OF_AGENTS
    num_episodes = MAX_NUM_EPISODES
    episode_steps = STEPS_PER_EPISODE
    env_params = EnvironmentParameters(
        rb_bandwidth, d2d_pair_distance, p_max, noise_power, bs_gain,
        user_gain, sinr_threshold_mue, n_mues, n_d2d, n_rb, bs_radius,
        c_param=C, mue_margin=mue_margin
    )
    agent_params = DQNAgentParameters(
        EPSILON_MIN, EPSILON_DECAY, 1, REPLAY_MEMORY_SIZE, BATCH_SIZE, GAMMA
    )
    reward_function = reward_func.dis_reward_tensor
    channel = BANChannel()
    env = CompleteEnvironment5(env_params, reward_function, channel)
    framework = ExternalDQNFramework(agent_params)
    framework.policy_net.load_state_dict(
        torch.load('models/dql/script31.pt')
    )
    mue_spectral_effs = [list() for _ in range(max_d2d+1)]
    d2d_spectral_effs = [list() for _ in range(max_d2d+1)]
    # jain_index = [list() for _ in range(max_d2d+1)]
    done = False
    bag = list()
    aux_range = range(max_d2d+1)[1:]
    for _ in range(num_episodes):
        # actions = np.linspace(1e-4, 1e-2, 5)[::-1] * p_max
        actions = np.linspace(1e-4, 8e-3, 5)[::-1] * p_max
        actions[0] = 0
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
                aux = agent.act(framework, obs[j]).max(1)
                agent.set_action(aux[1].long(), agent.actions[aux[1]])
                bag.append(aux[1].item())
            next_obs, rewards, done = env.step(agents)
            obs = next_obs
            total_reward += sum(rewards)
            i += 1
            if i >= episode_steps:
                break
        mue_spectral_effs[n_agents].append(env.mue_spectral_eff.item())
        d2d_spectral_effs[n_agents].append(env.d2d_spectral_eff.item())
        # jain_index[n_agents].append(gen.jain_index(env.sinr_d2ds))
    mue_success_rate = list()
    for i, m in enumerate(mue_spectral_effs):
        mue_success_rate.append(
            np.average(m > np.log2(1 + sinr_threshold_mue))
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
