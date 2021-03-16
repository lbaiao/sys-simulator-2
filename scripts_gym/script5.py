from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from sys_simulator import general as gen
import numpy as np
from time import time
from sys_simulator.a2c.agent import PPOAgent
from sys_simulator.a2c.framework import PPOFramework2
import torch
import gym
from sys_simulator.general.multiprocessing_env \
    import SubprocVecEnv, make_env

ALGO_NAME = 'ppo'
NUM_ENVS = 16
ENV_NAME = "Pendulum-v0"
HIDDEN_SIZE = 256
NUM_HIDDEN_LAYERS = 1
LEARNING_RATE = 3E-4
MAX_STEPS = 10000
STEPS_PER_EPISODE = 20
MINI_BATCH_SIZE = 5
PPO_EPOCHS = 4
THRESHOLD_REWARD = -150
BETA = .001
GAMMA = .99
LBDA = .95
CLIP_PARAM = .2
EVAL_NUM_EPISODES = 10
EVAL_EVERY = int(MAX_STEPS / 20)


torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = PPOAgent(torch_device)


def print_stuff(step: int, now: int):
    out = 'Training. ' + \
        f'Step: {step}/{MAX_STEPS-1}. ' + \
        f'Elapsed time: {now} minutes.'
    print(out)


def train(start: int, writer: SummaryWriter, timestamp: str,
          envs: SubprocVecEnv, framework: PPOFramework2, env: gym.Env):
    # writer.add_graph(framework.a2c.actor)
    # writer.add_graph(framework.a2c.critic)
    test_rewards = []
    step = 0
    early_stop = False
    while step < MAX_STEPS and not early_stop:
        obs = envs.reset()
        now = (time() - start) / 60
        print_stuff(step, now)
        reward = 0.0
        done = False
        t_flag = False
        total_entropy = 0
        for _ in range(STEPS_PER_EPISODE):
            action, log_prob, entropy, value = agent.act(obs, framework)
            next_obs, reward, done, _ = envs.step(action.cpu().numpy())
            total_entropy += entropy
            framework.push_experience(log_prob, value,
                                      reward, done, obs, action)
            obs = next_obs
            t_flag = True if step % EVAL_EVERY == 0 else t_flag
            step += 1
        if t_flag:
            t_rewards = test(framework, env)
            test_rewards.append(t_rewards)
            writer.add_scalar('mean reward', np.mean(t_rewards), step)
            if np.mean(t_rewards) > THRESHOLD_REWARD:
                early_stop = True
        _, _, _, next_value = agent.act(next_obs, framework)
        framework.push_next(next_obs, next_value, total_entropy)
        a_loss, c_loss = framework.learn()
        writer.add_scalar('actor loss', a_loss, step)
        writer.add_scalar('critic loss', c_loss, step)

    # last test
    t_rewards = test(framework)
    test_rewards.append(t_rewards)
    # save stuff
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    data_path = f'models/{ALGO_NAME}/gym/{filename}/{timestamp}'
    gen.make_dir(data_path)
    torch.save(framework, f'{data_path}/framework.pt')
    return test_rewards


def test(framework: PPOFramework2, env: gym.Env):
    rewards = []
    for _ in range(EVAL_NUM_EPISODES):
        obs = env.reset()
        done = False
        i = 0
        ep_rewards = []
        while not done and i < STEPS_PER_EPISODE:
            action, _, _, _ = agent.act(obs, framework)
            next_obs, reward, done, _ = env.step(action.cpu().numpy())
            obs = next_obs
            ep_rewards.append(reward)
        rewards.append(np.sum(ep_rewards))
    return rewards


def test_video(
    framework: PPOFramework2,
    num_episodes: int,
    steps_per_episode: int
):
    env = gym.make(f'{ENV_NAME}')
    for _ in range(num_episodes):
        obs = env.reset()
        env.render()
        done = False
        i = 0
        while not done and i < steps_per_episode:
            action, _, _, _ = agent.act(obs, framework)
            next_obs, _, done, _ = env.step(action.cpu().numpy())
            obs = next_obs
            env.render()


def exe(envs: SubprocVecEnv, framework: PPOFramework2, env: gym.Env):
    # make data dir
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    dir_path = f'data/{ALGO_NAME}/gym/{filename}'
    data_path, timestamp = gen.make_dir_timestamp(dir_path)
    writer = SummaryWriter(f'{data_path}/tensorboard')
    # train and test
    train_rewards = []
    test_rewards = []
    start = time()
    train_rewards = train(start, writer, timestamp, envs, framework, env)
    writer.close()
    test_rewards = test(framework, env)
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


def run():
    # env
    envs = [make_env(ENV_NAME) for _ in range(NUM_ENVS)]
    envs = SubprocVecEnv(envs)
    env = gym.make(ENV_NAME)
    # framework
    input_size = envs.observation_space.shape[0]
    output_size = envs.action_space.shape[0]
    framework = PPOFramework2(
        input_size,
        output_size,
        HIDDEN_SIZE,
        NUM_HIDDEN_LAYERS,
        STEPS_PER_EPISODE,
        NUM_ENVS,
        LEARNING_RATE,
        BETA,
        GAMMA,
        LBDA,
        CLIP_PARAM,
        torch_device
    )
    exe(envs, framework, env)


if __name__ == '__main__':
    run()
