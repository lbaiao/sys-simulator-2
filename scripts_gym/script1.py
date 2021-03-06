from sys_simulator.dqn.externalDQNFramework import RainbowFramework
from sys_simulator.parameters.parameters import AgentParameters
import gym

NUMBER = 2
STEPS_PER_EPISODE = 25
TEST_STEPS_PER_EPISODE = 25
MAX_NUM_EPISODES = 480 * NUMBER      # medium training
ITERATIONS_PER_NUM_AGENTS = 100
EVAL_EVERY = 20 * NUMBER
EVAL_NUM_EPISODES = 20
EVAL_STEPS_PER_EPISODE = 5
EPSILON_INITIAL = 1
EPSILON_MIN = .05
EPSILON_DECAY = 1.3/(MAX_NUM_EPISODES*STEPS_PER_EPISODE)  # medium training
PRIO_BETA_ITS = int(.8*MAX_NUM_EPISODES*STEPS_PER_EPISODE)
HIDDEN_SIZE = 256
NUM_HIDDEN_LAYERS = 1
LEARNING_RATE = 1e-3
TARGET_UPDATE = 24
REPLAY_MEMORY_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.5  # Discount factor


env = gym.make("CartPole-v1")
agent_params = AgentParameters(
    EPSILON_MIN, EPSILON_DECAY, EPSILON_INITIAL, REPLAY_MEMORY_SIZE,
    BATCH_SIZE, GAMMA
)
framework = RainbowFramework(
    agent_params,
    env.observation_space.shape[0],
    env.action_space.shape[0],
    HIDDEN_SIZE,
    PRIO_BETA_ITS,
    NUM_HIDDEN_LAYERS,
    LEARNING_RATE,
)


for _ in range(MAX_NUM_EPISODES):
    observation = env.reset()
    for _ in range(STEPS_PER_EPISODE):
        action = env.action_space.sample()  # your agent here (this takes random actions)  # noqa
        observation, reward, done, info = env.step(action)
        if done:
            break
env.render()
env.close()
