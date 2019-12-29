import sys
sys.path.insert(1, 'D:\\Dev\\sys-simulator-2')

from general import general as gen
from devices.devices import node, base_station, mobile_user, d2d_user, d2d_node_type
from pathloss import pathloss
from plots.plots import plot_positions

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
alpha = 0.5 # learning rate
etta = 0.9  # discount factor
epsilon = 0.1   # probability epsilon in epsilon-greedy
C = 80  # C constant for the improved reward function
# TODO: declarar ações e estados

# declaring the bs, mues and d2d pairs
bs = base_station((0,0), radius = bs_radius)
mues = [mobile_user(x) for x in range(n_mues)]
d2d_txs = [d2d_user(x, d2d_node_type.TX) for x in range(n_d2d)]
d2d_rxs = [d2d_user(x, d2d_node_type.RX) for x in range(n_d2d)]

# distributing nodes in the bs radius
gen.distribute_nodes(mues, bs)
for i in range(n_d2d):
    gen.distribute_pair_fixed_distance( list(zip(d2d_txs, d2d_rxs))[i], bs, d2d_pair_distance)

plot_positions(bs, mues, d2d_txs, d2d_rxs)



