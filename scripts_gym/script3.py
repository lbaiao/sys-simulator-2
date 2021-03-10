import gym
from sys_simulator.general.multiprocessing_env \
    import SubprocVecEnv, make_env

NUM_ENVS = 16
ENV_NAME = "Pendulum-v0"


envs = [make_env(ENV_NAME) for _ in range(NUM_ENVS)]
envs = SubprocVecEnv(envs)
env = gym.make(ENV_NAME)

