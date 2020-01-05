import sys

import os
lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from q_learning.environment import RLEnvironment

print('sucesso')