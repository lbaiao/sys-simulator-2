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

