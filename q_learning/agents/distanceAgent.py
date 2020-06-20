import os
import sys
# lucas_path = os.environ['LUCAS_PATH']
# sys.path.insert(1, lucas_path)

import numpy as np
from q_learning.q_table import QTable
from parameters.parameters import AgentParameters
from q_learning.agents.agent import Agent

class DistanceAgent(Agent):
    """
    don't forget to set the agent actions with the set_actions method
    """
    def __init__(self, params: AgentParameters, actions):
        super(DistanceAgent, self).__init__(params, actions)

    def set_distance_to_bs(self, distance: float):
        self.distance_to_bs = distance
        