# A2C script, but with completeEnvironment2. It uses multiple agents
# to train a single A2C network. The algorithm is trained with N_D2D
# varying from 1 to 10.

from sys_simulator.general import general as gen
from sys_simulator.q_learning.environments.completeEnvironmentA2C \
    import CompleteEnvironmentA2C
from sys_simulator.q_learning.rewards import dis_reward_tensor
from sys_simulator.parameters.parameters import EnvironmentParameters
from sys_simulator.a2c.agent import Agent
from sys_simulator.a2c.a2c import ActorCritic, compute_gae_returns
from torch import optim, nn
import torch
import os
import pickle
import random
# from copy import deepcopy


def run():
    # environment physical parameters
    n_mues = 1  # number of mues
    n_d2d = 2  # number of d2d pairs
    n_rb = n_mues  # number of RBs
    bs_radius = 500  # bs radius in m
    rb_bandwidth = 180*1e3  # rb bandwidth in Hz
    d2d_pair_distance = 50  # d2d pair distance in m
    p_max = 23  # max tx power in dBm
    noise_power = -116  # noise power per RB in dBm
    bs_gain = 17    # macro bs antenna gain in dBi
    user_gain = 4   # user antenna gain in dBi
    sinr_threshold_train = 6  # mue sinr threshold in dB for training
    mue_margin = .5e4
    # conversions from dB to pow
    p_max = p_max - 30
    p_max = gen.db_to_power(p_max)
    noise_power = noise_power - 30
    noise_power = gen.db_to_power(noise_power)
    bs_gain = gen.db_to_power(bs_gain)
    user_gain = gen.db_to_power(user_gain)
    sinr_threshold_train = gen.db_to_power(sinr_threshold_train)
    # ai training parameters
    STEPS_PER_EPISODE = 20
    MAX_NUM_EPISODES = 2700 * 1     # long training
    # C = 8000 # C constant for the improved reward function
    C = 80  # C constant for the improved reward function
    MAX_NUMBER_OF_AGENTS = 10
    HIDDEN_SIZE = 256
    LEARNING_RATE = .1
    # mu = 0.82*p_max/5/2000
    # std = mu/6
    mu = 0
    std = 0.1
    # torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # parameters classes initialization
    env_params = EnvironmentParameters(
        rb_bandwidth, d2d_pair_distance, p_max, noise_power,
        bs_gain, user_gain, sinr_threshold_train,
        n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin)
    # environment initialization
    reward_function = dis_reward_tensor
    environment = CompleteEnvironmentA2C(env_params, reward_function)
    # a2c initialization
    a2c = ActorCritic(6, 1, HIDDEN_SIZE, mu, std)
    actor_optimizer = optim.SGD(a2c.actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.SGD(a2c.critic.parameters(), lr=LEARNING_RATE)
    # training loop
    episode = 0
    d2d_spectral_effs = []
    mue_spectral_effs = []
    while episode < MAX_NUM_EPISODES:
        # entropy = 0
        aux_range = range(MAX_NUMBER_OF_AGENTS+1)[1:]
        n_agents = random.choice(aux_range)
        agents = [Agent() for _ in range(n_agents)]
        environment.build_scenario(agents)
        obs = [environment.get_state(a) for a in agents]
        log_probs = torch.zeros((n_agents, STEPS_PER_EPISODE)).to(device)
        values = torch.zeros((n_agents, STEPS_PER_EPISODE+1)).to(device)
        rewards = torch.zeros((n_agents, STEPS_PER_EPISODE)).to(device)
        i = 0
        done = False
        # actions = []  # used for debug purposes
        while not done and i < STEPS_PER_EPISODE:
            # agents choose their actions
            # actions_t = []  # used for debug purposes
            for j, agent in enumerate(agents):
                action, dist, value = agent.act(a2c, obs[j])
                # actions_t.append(action)    # used for debug purposes
                log_prob = dist.log_prob(action)
                # entropy += dist.entropy().mean()
                log_probs[j][i] = log_prob
                values[j][i] = value
            # perform a environment step
            next_obs_t, rewards_t, done = environment.step(agents)
            rewards[:, i] = torch.FloatTensor(rewards_t)
            # actions.append(actions_t)   # used for debug purposes
            i += 1
            # last_states = deepcopy(obs)  # used for debug purposes
            obs = next_obs_t
        # gae and returns
        next_obs_t = torch.cat(obs, 0).to(device)
        for j, agent in enumerate(agents):
            _, _, next_value_t = agents[0].act(a2c, next_obs_t[j])
            values[j][i] = next_value_t
        advantages, returns = compute_gae_returns(device, rewards, values)
        # update critic
        values_critic = values[:, :-1].reshape(1, -1).to(device)
        returns_critic = returns.view(1, -1).to(device)
        critic_loss = nn.functional.mse_loss(values_critic, returns_critic)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        # update actor
        aux = torch.mul(advantages, log_probs)
        aux = torch.sum(aux, axis=1)
        actor_loss = -torch.mean(aux) * 10
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # print training info
        episode += 1
        m_reward = torch.mean(rewards).item()
        d2d_spectral_effs.append(environment.d2d_spectral_eff)
        mue_spectral_effs.append(environment.mue_spectral_eff)
        print("Episode#:{} mean reward:{}".format(
            episode, m_reward))
    # save training data into a file
    cwd = os.getcwd()
    data = {}
    data['d2d_spectral_effs'] = d2d_spectral_effs
    data['mue_spectral_effs'] = mue_spectral_effs
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    filename_model = filename
    filename = f'{cwd}/data/a2c/{filename}.pickle'
    # save the a2c models
    torch.save(
        a2c.state_dict(),
        f'{cwd}/models/a2c/{filename_model}.pt')
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
