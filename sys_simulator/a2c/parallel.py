import numpy as np


def step(args):
    env = args[0]
    agents = args[1]
    a, b, c, d = env.step(agents)
    c = c * np.ones(len(agents))
    return a, b, c, d, env, agents


def unpack_multi_agent(x, n_envs, n_agents):
    total = n_envs * n_agents
    states, rewards, dones, _, envs, agents = zip(*x)
    states = np.array(states).reshape(total, -1)
    rewards = np.array(rewards).reshape(total, -1)
    dones = np.array(dones).reshape(total, -1)
    return states, rewards, dones, envs, agents


def unpack_multi_agent_test(x, n_envs, n_agents):
    total = n_envs * n_agents
    states, rewards, dones, _, envs, agents = zip(*x)
    states = np.array(states).reshape(total, -1)
    rewards = np.array(rewards)
    dones = np.array(dones).reshape(total, -1)
    return states, rewards, dones, envs, agents


def env_step(pool, envs, agents):
    n_envs = len(envs)
    n_agents = len(agents[0])
    aux = pool.map(step, zip(envs, agents))
    next_obs, reward, done, envs, agents = \
        unpack_multi_agent(aux, n_envs, n_agents)
    return next_obs, reward, done, envs, agents


def env_step_test(pool, envs, agents):
    n_envs = len(envs)
    n_agents = len(agents[0])
    aux = pool.map(step, zip(envs, agents))
    next_obs, reward, done, envs, agents = \
        unpack_multi_agent_test(aux, n_envs, n_agents)
    return next_obs, reward, done, envs, agents
