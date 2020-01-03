import os
import sys
lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)
# sys.path.insert(1, 'D:\\Dev\\sys-simulator-2')

import numpy as np
from q_learning.environment import Environment
from q_learning.q_table import QTable

class Agent:
    def __init__(self, env: Environment, max_episodes: float, steps_per_episode: float, epsilon_min: float,
                    epsilon_decay: float, alpha: float, gamma: float):
        self.env = env
        self.max_episodes = max_episodes
        self.steps_per_episode = steps_per_episode
        self.max_steps = max_episodes * steps_per_episode
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = 1
        self.alpha = alpha
        self.gamma = gamma
        self.actions = env.get_actions()        

    def set_q_table(self, q_table: QTable):
        self.q_table = q_table

    def get_action(self, obs):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() > self.epsilon_decay:
            return np.argmax(self.q_table.table[obs,:])
        else:
            return np.random.choice(self.actions)
        