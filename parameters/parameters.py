class EnvironmentParameters:
    def __init__(self, rb_bandwidth: float, d2d_pair_distance: float, p_max: float, noise_power: float, bs_gain: float, user_gain: float, sinr_threshold: float,
                    n_mues: int, n_d2d: int, n_rb: int, bs_radius: float):
        self.rb_bandwidth = rb_bandwidth
        self.d2d_pair_distance = d2d_pair_distance        
        self.p_max = p_max
        self.noise_power = noise_power
        self.bs_gain = bs_gain
        self.user_gain = user_gain
        self.sinr_threshold = sinr_threshold
        self.n_mues = n_mues
        self.n_d2d = n_d2d
        self.n_rb = n_rb
        self.bs_radius = bs_radius               


class TrainingParameters:
    def __init__(self, max_episodes: int, steps_per_episode: int, max_steps: int):  
        self.max_episodes = max_episodes
        self.steps_per_episode = steps_per_episode
        self.max_steps = max_steps


class AgentParameters:
    def __init__(self, epsilon_min, epsilon_decay, start_epsilon):
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.start_epsilon = start_epsilon


class LearningParameters:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma