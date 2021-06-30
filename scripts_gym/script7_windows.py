from sys_simulator.a2c.agent import A2CAgent
from sys_simulator.a2c.framework import A2CDiscreteFramework
import torch
import gym

ALGO_NAME = 'a2c'
NUM_ENVS = 8
ENV_NAME = 'CartPole-v1'
HIDDEN_SIZE = 256
NUM_HIDDEN_LAYERS = 1
LEARNING_RATE = 3E-4
MAX_STEPS = 200000
STEPS_PER_EPISODE = 300
THRESHOLD_REWARD = 450
BETA = .001
GAMMA = .99
LBDA = .95
EVAL_NUM_EPISODES = 10
EVAL_EVERY = int(MAX_STEPS / 20)


torch_device = torch.device("cpu")
agent = A2CAgent(torch_device)


def test_video(
    framework: A2CDiscreteFramework,
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
            next_obs, _, done, _ = env.step(action.item())
            obs = next_obs
            env.render()
