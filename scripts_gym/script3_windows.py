from sys_simulator.a2c.agent import PPOAgent
from sys_simulator.a2c.framework import PPOFramework
import torch
import gym


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
THRESHOLD_REWARD = -250
BETA = .001
GAMMA = .99
LBDA = .95
CLIP_PARAM = .2
EVAL_NUM_EPISODES = 10
EVAL_EVERY = int(MAX_STEPS / 20)


torch_device = torch.device("cpu")
# env
env = gym.make(ENV_NAME)
# framework
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
framework = PPOFramework(
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
agent = PPOAgent(torch_device)


def test_video(
    framework: PPOFramework,
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
