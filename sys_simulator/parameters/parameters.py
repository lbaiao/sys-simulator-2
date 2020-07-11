class EnvironmentParameters:
    def __init__(self, rb_bandwidth: float, d2d_pair_distance: float, p_max: float, noise_power: float, bs_gain: float, user_gain: float, sinr_threshold: float,
                    n_mues: int, n_d2d: int, n_rb: int, bs_radius: float, **kwargs):
        self.rb_bandwidth = rb_bandwidth
        self.d2d_pair_distance = d2d_pair_distance        
        self.p_max = p_max
        self.noise_power = noise_power
        self.bs_gain = bs_gain
        self.user_gain = user_gain
        self.sinr_threshold = sinr_threshold
        self.n_mues = int(n_mues)
        self.n_d2d = int(n_d2d)
        self.n_rb = int(n_rb)
        self.bs_radius = bs_radius               

        if 'c_param' in kwargs:
            self.c_param = kwargs['c_param']

        if 'mue_margin' in kwargs:
            self.mue_margin = kwargs['mue_margin']


class TrainingParameters:
    def __init__(self, max_episodes: int, steps_per_episode: int):  
        self.max_episodes = int(max_episodes)
        self.steps_per_episode = int(steps_per_episode)


class AgentParameters:
    def __init__(self, epsilon_min, epsilon_decay, start_epsilon):
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.start_epsilon = start_epsilon

class DQNAgentParameters(AgentParameters):
    def __init__(self, epsilon_min, epsilon_decay, start_epsilon, replay_memory_size, batchsize, gamma):
        super(DQNAgentParameters, self).__init__(epsilon_min, epsilon_decay, start_epsilon)
        self.batchsize = batchsize
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size


class LearningParameters:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma