import sys
import os

lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from general import general as gen
from devices.devices import node, base_station, mobile_user, d2d_user, d2d_node_type
from pathloss import pathloss
from plots.plots import plot_positions, plot_spectral_effs
from q_learning.environments.distributedEnvironment import DistributedEnvironment
from q_learning.agents.agent import Agent
from q_learning.q_table import DistributedQTable
from q_learning import rewards
from parameters.parameters import EnvironmentParameters, TrainingParameters, AgentParameters, LearningParameters
from typing import List
from sinr.sinr import sinr_d2d, sinr_mue


import math
import numpy as np

x = 7 - 30
y = gen.db_to_power(x)
if y == 0.005011872336272725:
    print('PASS gen.db_to_power')
else:
    print('FAIL gen.db_to_power')


y = pathloss.pathloss_users(50/1000)
if y == 0.003943483403001204:
    print('PASS pathloss_users')
else:
    print('FAIL pathloss_users')


y = pathloss.pathloss_bs_users(200/1000)
if y == 0.0797760967818817:
    print('PASS pathloss_bs_users')
else:
    print('FAIL pathloss_bs_users')


bs = base_station((0,0))
mue = mobile_user(1)
mue.set_position((100,0))
mue.set_tx_power(1.9953e-1)
mue.set_distance_to_bs(100)
mue.set_rb(1)
d2d1_1 = d2d_user(1, d2d_node_type.TX)
d2d1_1.set_position((-100,0))
d2d1_1.set_distance_d2d(50)
d2d1_1.set_rb(1)
d2d1_1.set_tx_power(3.1623e-3)
d2d1_1.set_distance_to_bs(100)
d2d1_2 = d2d_user(1, d2d_node_type.RX)
d2d1_2.set_position((-150,0))
d2d1_2.set_distance_d2d(50)
d2d1_2.set_rb(1)
d2d2_1 = d2d_user(2, d2d_node_type.TX)
d2d2_1.set_position((0,200))
d2d2_1.set_distance_d2d(50)
d2d2_1.set_rb(1)
d2d2_1.set_tx_power(5.0812e-3)
d2d1_1.set_distance_to_bs(200)
d2d2_2 = d2d_user(2, d2d_node_type.RX)
d2d2_2.set_position((0,150))
d2d2_2.set_distance_d2d(50)
d2d2_2.set_rb(1)
d2d_users = [d2d1_1, d2d1_2, d2d2_1, d2d2_2]


sinr1 = sinr_d2d(d2d1_1, d2d1_2, d2d_users, mue, 2.5119e-15, 2.5119)
if sinr1 == 9.659478562268228:
    print('PASS sinr_d2d')
else:
    print('DONE sinr_d2d')


sinrm = sinr_mue(mue, d2d_users, bs, 2.5119e-15, 50.1187, 2.5119)
if sinrm == 56.406572374295244:
    print('PASS sinr_mue')
else:
    print('DONE sinr_mue')




