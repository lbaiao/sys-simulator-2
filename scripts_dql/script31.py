# Same as script 23, but with the BAN channel.
from sys_simulator.channels import BANChannel
from sys_simulator.general import general as gen
from sys_simulator.q_learning.environments.completeEnvironment5 \
    import CompleteEnvironment5
from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from sys_simulator.dqn.externalDQNFramework import ExternalDQNFramework
from sys_simulator.parameters.parameters import \
    EnvironmentParameters, TrainingParameters, DQNAgentParameters
from sys_simulator.q_learning import rewards as reward_functions
import torch
import numpy as np
import os


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
    sinr_threshold_train = 6  # mue sinr threshold in dB for training
    mue_margin = 2  # mue margin in dB
    # conversions from dB to pow
    p_max = p_max - 30
    p_max = gen.db_to_power(p_max)
    noise_power = noise_power - 30
    noise_power = gen.db_to_power(noise_power)
    bs_gain = gen.db_to_power(bs_gain)
    user_gain = gen.db_to_power(user_gain)
    sinr_threshold_train = gen.db_to_power(sinr_threshold_train)
    mue_margin = gen.db_to_power(mue_margin)
    # q-learning parameters
    STEPS_PER_EPISODE = 25
    MAX_NUM_EPISODES = 480      # medium training
    # MAX_NUM_EPISODES = 1      # testing
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 3.35*1e-4    # medium training
    GAMMA = 0.98  # Discount factor
    C = 80  # C constant for the improved reward function
    TARGET_UPDATE = 10
    MAX_NUMBER_OF_AGENTS = 10
    max_d2d = MAX_NUMBER_OF_AGENTS
    # more parameters
    env_params = EnvironmentParameters(
        rb_bandwidth, d2d_pair_distance, p_max, noise_power,
        bs_gain, user_gain, sinr_threshold_train,
        n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin
    )
    params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
    agent_params = DQNAgentParameters(
        EPSILON_MIN, EPSILON_DECAY, 1, 10000, 512, GAMMA
    )
    framework = ExternalDQNFramework(agent_params)
    reward_function = reward_functions.dis_reward_tensor
    channel = BANChannel()
    env = CompleteEnvironment5(env_params, reward_function, channel)
    best_reward = float('-inf')
    device = torch.device('cuda')
    mue_spectral_eff_bag = list()
    d2d_spectral_eff_bag = list()
    aux_range = range(max_d2d+1)[1:]
    epsilon = agent_params.start_epsilon
    for episode in range(params.max_episodes):
        actions = [i*0.82*p_max/5/1000 for i in range(5)]  # best result
        n_agents = np.random.choice(aux_range)
        agents = [ExternalDQNAgent(agent_params, actions)
                  for _ in range(n_agents)]  # 1 agent per d2d tx
        counts = np.zeros(len(agents))
        awaits = list()
        await_steps = [2, 3, 4]
        for a in agents:
            awaits.append(np.random.choice(await_steps))
            a.set_action(torch.tensor(0).long().cuda(), a.actions[0])
            a.set_epsilon(epsilon)
        env.build_scenario(agents)
        done = False
        obs = [env.get_state(a) for a in agents]
        total_reward = 0.0
        i = 0
        bag = list()
        while not done:
            if i >= params.steps_per_episode:
                break
            else:
                actions = torch.zeros([len(agents)], device=device)
                for j, agent in enumerate(agents):
                    if counts[j] < awaits[j]:
                        counts[j] += 1
                    else:
                        agent.get_action(framework, obs[j])
                        actions[j] = agent.action_index
                        counts[j] = 0
                        awaits[j] = np.random.choice(await_steps)
                next_obs, rewards, done = env.step(agents)
                i += 1
                for j, agent in enumerate(agents):
                    framework.replay_memory.push(obs[j], actions[j],
                                                 next_obs[j], rewards[j])
                framework.learn()
                obs = next_obs
                total_reward += torch.sum(rewards)
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
            mue_spectral_eff_bag.append(env.mue_spectral_eff)
            # average d2d spectral eff
            d2d_spectral_eff_bag.append(env.d2d_spectral_eff/env.params.n_d2d)
        epsilon = agents[0].epsilon
    # Return the trained policy
    mue_spectral_effs = mue_spectral_eff_bag
    d2d_spectral_effs = d2d_spectral_eff_bag
    spectral_effs = zip(mue_spectral_effs, d2d_spectral_effs)
    # saving the data and the model
    cwd = os.getcwd()
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    filename_model = filename
    filename = f'{cwd}/data/dql/{filename}.pt'
    torch.save(framework.policy_net.state_dict(),
               f'{cwd}/models/dql/{filename_model}.pt')
    torch.save(spectral_effs, filename)
