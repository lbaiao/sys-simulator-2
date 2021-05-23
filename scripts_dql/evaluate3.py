from shutil import copyfile
from typing import List
from time import time
from sys_simulator.general.actions_discretizations import db_six, db_ten
from sys_simulator.q_learning.environments.completeEnvironment12 import CompleteEnvironment12
import sys_simulator.general as gen
from sys_simulator.general import load_with_pickle, print_evaluate3, random_seed, save_with_pickle
from sys_simulator.parameters.parameters import DQNAgentParameters
from sys_simulator.dqn.externalDQNFramework import ExternalDQNFramework
from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
import torch


# parameters
ALGO_NAME = 'dql'
BASE_PATH = '/home/lucas/dev/sys-simulator-2'
AGENTS_RANGE = range(6)[1:]
MODELS_PATHS = [
    f'{BASE_PATH}/data/dql/script52/20210515-172025/last_model.pt',
    f'{BASE_PATH}/data/dql/script52/20210515-143418/last_model.pt',
    f'{BASE_PATH}/data/dql/script52/20210510-201634/last_model.pt',
    f'{BASE_PATH}/data/dql/script52/20210516-103139/best_model.pt',
    f'{BASE_PATH}/data/dql/script52/20210516-113258/best_model.pt',
]
ENVS_PATHS = [
    f'{BASE_PATH}/data/dql/script52/20210515-172025/env.pickle',
    f'{BASE_PATH}/data/dql/script52/20210515-143418/env.pickle',
    f'{BASE_PATH}/data/dql/script52/20210510-201634/env.pickle',
    f'{BASE_PATH}/data/dql/script52/20210516-103139/env.pickle',
    f'{BASE_PATH}/data/dql/script52/20210516-113258/env.pickle',
]
# TEST_NUM_EPISODES = 10000
TEST_NUM_EPISODES = 1000
EVAL_STEPS_PER_EPISODE = 10
PRINT_EVERY = 100
# env parameters
RND_SEED = True
SEED = 42
CHANNEL_RND = True
# writer
filename = gen.path_leaf(__file__)
filename = filename.split('.')[0]
dir_path = f'data/{ALGO_NAME}/{filename}'
data_path, _ = gen.make_dir_timestamp(dir_path)
if RND_SEED:
    random_seed(SEED)
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a_min = -90
a_max = 60
a_offset = -10
frameworks = []
for p in MODELS_PATHS:
    f = torch.load(p, map_location=torch_device)
    f.policy_net.to(torch_device)
    f.device = torch_device
    frameworks.append(f)
# envs = [load_with_pickle(p) for p in ENVS_PATHS]
p_min = -90
p_max = 40  # max tx power in dBm
p_max = p_max - 30
actions_six = db_six(p_min, p_max)
actions_ten = db_ten(p_min, p_max)
actions = [actions_six for _ in range(5)]
# actions.append(actions_ten)
#
agent_params = DQNAgentParameters(1, 1, 1, 1, 1, 1)
envs = [load_with_pickle(e) for e in ENVS_PATHS]

def test(framework: ExternalDQNFramework, env: CompleteEnvironment12,
         surr_agents: List[ExternalDQNAgent], start: float):
    framework.policy_net.eval()
    mue_availability = []
    mue_sinrs = []
    d2d_sinrs = []
    rewards_bag = []
    for ep in range(TEST_NUM_EPISODES):
        if ep % PRINT_EVERY == 0:
            now = (time() - start) / 60
            print_evaluate3(ep, TEST_NUM_EPISODES, now, len(surr_agents))
        env.reset()
        env.build_scenario(surr_agents, motion_model='random')
        obs, _, _, _ = env.step(surr_agents)
        i = 0
        done = False
        ep_availability = []
        ep_rewards = []
        ep_mue_sinrs = []
        ep_d2d_sinrs = []
        while not done and i < EVAL_STEPS_PER_EPISODE:
            # actions = np.zeros(MAX_NUMBER_OF_AGENTS) + 1e-9
            # db_actions = power_to_db(actions)
            for j, agent in enumerate(agents):
                agent.act(framework, obs[j])
            next_obs, reward, done, _ = env.step(surr_agents)
            obs = next_obs
            ep_availability.append(env.mue.sinr > env.params.sinr_threshold)
            ep_rewards.append(reward)
            ep_mue_sinrs.append(env.mue.sinr)
            ep_d2d_sinrs.append([p[0].sinr for p in env.d2d_pairs])
            i += 1
        rewards_bag += ep_rewards
        mue_sinrs += ep_mue_sinrs
        d2d_sinrs += ep_d2d_sinrs
        mue_availability += ep_availability
    all_bags = {
        'rewards': rewards_bag,
        'mue_sinrs': mue_sinrs,
        'd2d_sinrs': d2d_sinrs,
        'mue_availability': mue_availability
    }
    return all_bags


if __name__ == '__main__':
    filename = gen.path_leaf(__file__)
    filename = filename.split('.')[0]
    dir_path = f'data/{ALGO_NAME}/{filename}'
    data_path, _ = gen.make_dir_timestamp(dir_path)
    start = time()
    results = []
    for f, i, e, a in zip(frameworks, AGENTS_RANGE, envs, actions):
        agents = [
            ExternalDQNAgent(agent_params, a)
            for _ in range(i)
        ]
        r = test(f, e, agents, start)
        results.append(r)
    # save stuff
    now = (time() - start) / 60
    data_file_path = f'{data_path}/log.pickle'
    save_with_pickle(results, data_file_path)
    copyfile(__file__, f'{data_path}/{filename}.py')
    print(f'done. Elapsed time: {now} minutes.')

