import sys
import os

lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

# sys.path.insert(1, 'D:\\Dev\\sys-simulator-2')

from general import general as gen
from devices.devices import node, base_station, mobile_user, d2d_user, d2d_node_type
from pathloss import pathloss
from plots.plots import plot_positions
from q_learning.environment import Environment, EnvironmentParameters
from q_learning.agent import Agent

import math

n_mues = 1 # number of mues
n_d2d = 3  # number of d2d pairs
n_rb = n_mues   # number of RBs
bs_radius = 500 #   bs radius in m

rb_bandwidth = 180*1e3  # rb bandwidth in Hz
d2d_pair_distance = 50  # d2d pair distance in m
p_max = 23  # max tx power in dBm
noise_power = -116  # noise power per RB in dBm
bs_gain = 17    # macro bs antenna gain in dBi
user_gain = 4   # user antenna gain in dBi
sinr_threshold = 6  # mue sinr threshold in dB

# conversions to dB
p_max = p_max - 30
p_max = gen.db_to_power(p_max)
noise_power = noise_power - 30
noise_power = gen.db_to_power(noise_power)
bs_gain = gen.db_to_power(bs_gain)
user_gain = gen.db_to_power(user_gain)
sinr_threshold = gen.db_to_power(sinr_threshold)

# q-learning parameters
MAX_NUM_EPISODES = 20000
STEPS_PER_EPISODE = 200 
EPSILON_MIN = 0.1
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 1000 * EPSILON_MIN / max_num_steps
ALPHA = 0.5  # Learning rate
GAMMA = 0.9  # Discount factor
C = 80  # C constant for the improved reward function
parameters = EnvironmentParameters(rb_bandwidth, d2d_pair_distance, p_max, noise_power, bs_gain, user_gain, sinr_threshold,
                                        n_mues, n_d2d, n_rb, bs_radius)
environment = Environment(parameters)

# training function
# TODO: colocar agente e d2d_device na mesma classe? fazer propriedade d2d_device no agente?
def train(agent: Agent, env: Environment):
    best_reward = 0
    for episode in range(agent.max_episodes):
        # TODO: atualmente redistribuo os usuarios aleatoriamente a cada episodio. Isto é o melhor há se fazer? 
        # Simular deslocamento dos usuários?
        env.reset()
        done = False
        obs = env.get_state()
        total_reward = 0.0
        i = 0
        while not done:
            if i >= agent.max_steps:
                break
            else:
                action = agent.get_action(obs)

        







