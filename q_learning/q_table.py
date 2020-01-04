import sys
import os
lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

import numpy as np
from q_learning.environment import Environment

class QTable:
    def __init__(self, env: Environment):
        self.table = np.zeros((2, len(env.get_actions())))