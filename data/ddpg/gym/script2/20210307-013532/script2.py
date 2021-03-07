from shutil import copyfile
from sys_simulator.ddpg.agent import Agent
import torch
from sys_simulator.ddpg.framework import Framework
import gym
from time import time
import sys_simulator.general as gen
import numpy as np

MAX_NUM_EPISODES = 4000
STEPS_PER_EPISODE = 1000
REPLAY_INITIAL = 10000
# MAX_NUM_EPISODES = 400
# STEPS_PER_EPISODE = 100
# REPLAY_INITIAL = 1000
EVAL_NUM_EPISODES = 10
REPLAY_MEMORY_SIZE = 100000
LEARNING_RATE = 1E-4
BATCH_SIZE = 64
GAMMA = .99
POLYAK = .999
ALPHA = .6
BETA = .4
PRIO_BETA_ITS = int(.8*(MAX_NUM_EPISODES*STEPS_PER_EPISODE - REPLAY_INITIAL))
EVAL_EVERY = int(MAX_NUM_EPISODES / 20)


torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make('Pendulum-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
a_min = env.action_space.low
a_max = env.action_space.high
framework = Framework(
    REPLAY_MEMORY_SIZE,
    REPLAY_INITIAL,
    state_size,
    action_size,
    LEARNING_RATE,
    BATCH_SIZE,
    GAMMA,
    POLYAK,
    PRIO_BETA_ITS,
    ALPHA,
    BETA,
    torch_device
)
agent = Agent(a_min, a_max, torch_device)


def train(start):
    best_reward = float('-inf')
    test_rewards = []
    for episode in range(MAX_NUM_EPISODES):
        obs = env.reset()
        now = (time() - start) / 60
        print(
            'Training. ' +
            f'Episode: {episode}/{MAX_NUM_EPISODES-1}. ' +
            f'Prio_Beta: {framework.replay_memory._beta}. ' +
            f'Elapsed time: {now} minutes.'
        )
        reward = 0.0
        done = False
        i = 0
        while not done:
            if i >= STEPS_PER_EPISODE:
                break
            action = agent.act(obs, framework, True)
            next_obs, reward, done, _ = env.step(action)
            framework.replay_memory.push(
                obs, action, reward, next_obs, done
            )
            i += 1
            framework.learn()
            best_reward = reward if reward > best_reward else best_reward
        if episode % EVAL_EVERY == 0:
            t_rewards = test(framework)
            test_rewards.append(t_rewards)
        framework.replay_memory.correct_beta(i, STEPS_PER_EPISODE)
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
        rewards.append(np.mean(ep_rewards))
    return rewards


def test_video(
    framework: Framework,
    num_episodes: int,
    steps_per_episode: int
):
    env = gym.make('Pendulum-v0')
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
    train_rewards = []
    test_rewards = []
    start = time()
    train_rewards = train(start)
    test_rewards = test(framework)
    # save stuff
    now = (time() - start) / 60
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    dir_path = f'data/ddpg/gym/{filename}'
    data_path = gen.make_dir_timestamp(dir_path)
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
