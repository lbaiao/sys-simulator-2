from shutil import copyfile
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from sys_simulator.general.ou_noise import OUNoise
from sys_simulator.ddpg.agent import Agent
import torch
from sys_simulator.ddpg.framework import Framework
import gym
from time import time
import sys_simulator.general as gen

# MAX_NUM_EPISODES = 12000
# STEPS_PER_EPISODE = 500
# REPLAY_INITIAL = 10000
MAX_STEPS = 12000
STEPS_PER_EPISODE = 500
REPLAY_INITIAL = int(0E3)
EVAL_NUM_EPISODES = 10
REPLAY_MEMORY_SIZE = int(1E6)
ACTOR_LEARNING_RATE = 1E-4
CRITIC_LEARNING_RATE = 1E-3
HIDDEN_SIZE = 256
N_HIDDEN_LAYERS = 2
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
# env = NormalizedActions(gym.make('Pendulum-v0'))
env = gym.make('Pendulum-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
a_min = env.action_space.low
a_max = env.action_space.high
framework = Framework(
    REPLAY_MEMORY_TYPE,
    REPLAY_MEMORY_SIZE,
    REPLAY_INITIAL,
    state_size,
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
ou_noise = OUNoise(
    env.action_space,
    OU_MU, OU_THETA,
    OU_MAX_SIGMA,
    OU_MIN_SIGMA,
    OU_DECAY_PERIOD
)
agent = Agent(-1.0, 1.0, EXPLORATION, torch_device)


def print_stuff(step: int, now: int):
    if REPLAY_MEMORY_TYPE == 'prioritized':
        out = 'Training. ' + \
            f'Step: {step}/{MAX_STEPS-1}. ' + \
            f'Prio_Beta: {framework.replay_memory._beta}. ' + \
            f'Elapsed time: {now} minutes.'
    else:
        out = 'Training. ' + \
            f'Step: {step}/{MAX_STEPS-1}. ' + \
            f'Elapsed time: {now} minutes.'
    print(out)


def train(start, writer: SummaryWriter, timestamp: str):
    actor_losses_bag = list()
    critic_losses_bag = list()
    best_reward = float('-inf')
    test_rewards = []
    step = 0
    while step < MAX_STEPS:
        obs = env.reset()
        ou_noise.reset()
        now = (time() - start) / 60
        print_stuff(step, now)
        reward = 0.0
        done = False
        i = 0
        while not done and i < STEPS_PER_EPISODE:
            action = agent.act(obs, framework, True, ou=ou_noise, step=i)
            next_obs, reward, done, _ = env.step(action)
            framework.replay_memory.push(
                obs, action, reward, next_obs, done
            )
            actor_loss, critic_loss = framework.learn()
            writer.add_scalar('Actor Losses', actor_loss, step)
            writer.add_scalar('Critic Losses', critic_loss, step)
            best_reward = reward if reward > best_reward else best_reward
            obs = next_obs
            i += 1
            step += 1
        if step % EVAL_EVERY == 0:
            t_rewards = test(framework)
            test_rewards.append(t_rewards)
            writer.add_scalar('Avg test rewards', np.mean(t_rewards), step)
        # if REPLAY_MEMORY_TYPE == 'prioritized':
        #     framework.replay_memory.correct_beta(i, STEPS_PER_EPISODE)
    # last test
    t_rewards = test(framework)
    test_rewards.append(t_rewards)
    # save stuff
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    data_path = f'models/ddpg/gym/{filename}'
    data_path = gen.make_dir_timestamp(data_path)
    torch.save(framework, f'{data_path}/framework.pt')
    return test_rewards


def test(framework: Framework):
    rewards = []
    for _ in range(EVAL_NUM_EPISODES):
        obs = env.reset()
        done = False
        i = 0
        ep_rewards = []
        while not done and i < STEPS_PER_EPISODE:
            action = agent.act(obs, framework, False)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            ep_rewards.append(reward)
        # rewards.append(np.mean(ep_rewards))
        rewards.append(np.sum(ep_rewards))
    return rewards


def test_video(
    framework: Framework,
    num_episodes: int,
    steps_per_episode: int
):
    env = gym.make('Pendulum-v0')
    agent = Agent(env.action_space.low,
                  env.action_space.high, EXPLORATION, torch_device)
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        i = 0
        while not done and i < steps_per_episode:
            env.render()
            action = agent.act(obs, framework, False)
            next_obs, _, done, _ = env.step(action)
            obs = next_obs


def run():
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    dir_path = f'data/ddpg/gym/{filename}'
    data_path, timestamp = gen.make_dir_timestamp(dir_path)
    writer = SummaryWriter(f'{data_path}/tensorboard')
    train_rewards = []
    test_rewards = []
    start = time()
    train_rewards = train(start, writer, timestamp)
    writer.close()
    test_rewards = test(framework)
    # save stuff
    now = (time() - start) / 60
    data_file_path = f'{data_path}/log.pickle'
    data = {
        'train_rewards': train_rewards,
        'test_rewards': test_rewards,
        'elapsed_time': now,
        'eval_every': EVAL_EVERY,
    }
    gen.save_with_pickle(data, data_file_path)
    copyfile(__file__, f'{data_path}/{filename}.py')
    print(f'done. Elapsed time: {now} minutes.')


if __name__ == '__main__':
    run()
