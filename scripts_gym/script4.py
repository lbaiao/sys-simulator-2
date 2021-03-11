from shutil import copyfile
import numpy as np
import sys_simulator.general as gen
from time import time
from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from sys_simulator.dqn.externalDQNFramework import ExternalDQNFramework
from sys_simulator.parameters.parameters import DQNAgentParameters
import torch
import gym


# ENV_NAME = 'CartPole-v1'
ENV_NAME = 'MountainCar-v0'
MAX_STEPS = 20000
STEPS_PER_EPISODE = 300
EVAL_NUM_EPISODES = 10
REPLAY_MEMORY_TYPE = 'prioritized'
REPLAY_MEMORY_SIZE = int(1E5)
ALPHA = .6
BETA = .4
PRIO_BETA_ITS = int(.8*MAX_STEPS)
LEARNING_RATE = 1E-3
HIDDEN_SIZE = 256
BATCH_SIZE = 128
GAMMA = .98
EPSILON_INITIAL = 1
EPSILON_MIN = .02
EPSILON_DECAY = 1.2/(MAX_STEPS)  # medium training
TARGET_UPDATE = 20
EVAL_EVERY = int(MAX_STEPS / 20)


torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make(ENV_NAME)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent_params = DQNAgentParameters(
    EPSILON_MIN, EPSILON_DECAY, EPSILON_INITIAL, REPLAY_MEMORY_SIZE,
    BATCH_SIZE, GAMMA
)
framework = ExternalDQNFramework(
    agent_params,
    state_size,
    action_size,
    HIDDEN_SIZE,
    torch_device,
    n_hidden_layers=2,
    learning_rate=LEARNING_RATE,
    alpha=ALPHA,
    beta=BETA,
    beta_its=PRIO_BETA_ITS,
    replay_memory_type=REPLAY_MEMORY_TYPE
)
agent = ExternalDQNAgent(agent_params, list(range(action_size)))


def print_stuff(step: int, now: int):
    if REPLAY_MEMORY_TYPE == 'prioritized':
        out = 'Training. ' + \
            f'Step: {step}/{MAX_STEPS-1}. ' + \
            f'Prio_Beta: {framework.replay_memory._beta}. ' + \
            f'Elapsed time: {now} minutes.'
    else:
        out = 'Training. ' + \
            f'Step: {step}/{MAX_STEPS-1}. ' + \
            f'Epsilon: {agent.epsilon}' + \
            f'Elapsed time: {now} minutes.'
    print(out)


def train(start):
    best_reward = float('-inf')
    test_rewards = []
    step = 0
    while step < MAX_STEPS:
        obs = env.reset()
        now = (time() - start) / 60
        print_stuff(step, now)
        reward = 0.0
        done = False
        t_flag = False
        i = 0
        while not done and i < STEPS_PER_EPISODE:
            action = agent.get_action(framework, obs)
            next_obs, reward, done, _ = env.step(action)
            framework.replay_memory.push(
                obs, action, reward, next_obs, done
            )
            framework.learn()
            best_reward = reward if reward > best_reward else best_reward
            obs = next_obs
            t_flag = True if step % EVAL_EVERY == 0 else t_flag
            i += 1
            step += 1
            if step % TARGET_UPDATE == 0:
                framework.target_net.load_state_dict(
                    framework.policy_net.state_dict()
                )
        if t_flag:
            t_rewards = test(framework)
            test_rewards.append(t_rewards)
            t_flag = False
    # last test
    t_rewards = test(framework)
    test_rewards.append(t_rewards)
    # save stuff
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    data_path = f'models/dql/gym/{filename}'
    data_path = gen.make_dir_timestamp(data_path)
    torch.save(framework, f'{data_path}/framework.pt')
    return test_rewards


def test(framework: ExternalDQNFramework):
    rewards = []
    for _ in range(EVAL_NUM_EPISODES):
        obs = env.reset()
        done = False
        i = 0
        ep_rewards = []
        while not done and i < STEPS_PER_EPISODE:
            action = agent.act(framework, obs)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            ep_rewards.append(reward)
        # rewards.append(np.mean(ep_rewards))
        rewards.append(np.sum(ep_rewards))
    return rewards


def test_video(
    framework: ExternalDQNFramework,
    num_episodes: int,
    steps_per_episode: int
):
    env = gym.make(ENV_NAME)
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        i = 0
        while not done and i < steps_per_episode:
            env.render()
            action = agent.act(framework, obs)
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
    dir_path = f'data/dql/gym/{filename}'
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
