# A2C script, but with completeEnvironment2. It uses multiple agents
# to train a single A2C network. The algorithm is trained with N_D2D
# varying from 1 to 10.

import os
import sys

lucas_path = os.getcwd()
sys.path.insert(1, lucas_path)

from general import general as gen
from q_learning.environments.completeEnvironmentA2C \
    import CompleteEnvironmentA2C
from q_learning import rewards
from parameters.parameters import EnvironmentParameters,\
    TrainingParameters
from a2c.agent import Agent
from a2c.a2c import ActorCritic, compute_returns

import torch
from torch import optim
import pickle
import random

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
sinr_threshold_mue = 6  # true mue sinr threshold in dB
mue_margin = .5e4

# conversions from dB to pow
p_max = p_max - 30
p_max = gen.db_to_power(p_max)
noise_power = noise_power - 30
noise_power = gen.db_to_power(noise_power)
bs_gain = gen.db_to_power(bs_gain)
user_gain = gen.db_to_power(user_gain)
sinr_threshold_train = gen.db_to_power(sinr_threshold_train)

# q-learning parameters
STEPS_PER_EPISODE = 10
MAX_NUM_EPISODES = 2000      # long training
C = 80  # C constant for the improved reward function
MAX_NUMBER_OF_AGENTS = 10

HIDDEN_SIZE = 128
LEARNING_RATE = 3e-4
# mu = 0.82*p_max/5/2000
# std = mu/6
mu = p_max*1e-8
std = mu/100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# more parameters
env_params = EnvironmentParameters(
    rb_bandwidth, d2d_pair_distance, p_max, noise_power,
    bs_gain, user_gain, sinr_threshold_train,
    n_mues, n_d2d, n_rb, bs_radius, c_param=C, mue_margin=mue_margin)
train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
reward_function = rewards.dis_reward_tensor
environment = CompleteEnvironmentA2C(env_params, reward_function)
a2c = ActorCritic(6, 1, HIDDEN_SIZE, mu, std)
optimizer = optim.Adam(a2c.parameters(), lr=LEARNING_RATE)

episode = 0
mue_spectral_eff_bag = list()
d2d_spectral_eff_bag = list()
mean_rewards = []
while episode < MAX_NUM_EPISODES:
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0
    best_reward = float('-inf')
    device = torch.device('cuda')
    aux_range = range(MAX_NUMBER_OF_AGENTS+1)[1:]
    n_agents = random.choice(aux_range)
    agents = [Agent() for _ in range(n_agents)]
    environment.build_scenario(agents)
    obs = [environment.get_state(a) for a in agents]
    i = 0
    done = False
    while not done and i < STEPS_PER_EPISODE:
        actions = torch.zeros([len(agents)], device=device)

        for j, agent in enumerate(agents):
            action, dist, value = agent.act(a2c, obs[j])
            actions[j] = action
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)

        next_obs_t, rewards_t, done = environment.step(agents)
        rewards.append(torch.FloatTensor(rewards_t).unsqueeze(1).to(device))
        aux = (1 - done) * torch.ones(n_agents).to(device)
        masks.append(aux)
        i += 1
        obs = next_obs_t

    next_obs = torch.cat(next_obs_t, 0).to(device)
    next_value = list()
    for j, agent in enumerate(agents):
        _, _, next_value_t = agents[0].act(a2c, next_obs_t[j])
        next_value.append(next_value_t)
    next_value = torch.cat(next_value, 0).to(device)
    returns = compute_returns(next_value, rewards, masks)

    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    episode += 1

    m_reward = torch.mean(torch.cat(rewards)).item()
    mean_rewards.append(m_reward)
    print("Episode#:{} mean reward:{}".format(
        episode, m_reward))

cwd = os.getcwd()

data = {}
data['mean_rewards'] = mean_rewards

filename = gen.path_leaf(__file__)
filename = filename.split('.')[0]
filename_model = filename
filename = f'{cwd}/data/a2c/{filename}.pickle'
torch.save(
    a2c.state_dict(),
    f'{cwd}/models/a2c/{filename_model}.pt')
with open(filename, 'wb') as f:
    pickle.dump(data, f)
