import sys
sys.path.insert(1, 'D:\\Dev\\sys-simulator-2')

from devices.devices import base_station, mobile_user, d2d_user, d2d_node_type
from general import general as gen


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


class Environment:
    def __init__(self, params: EnvironmentParameters):
        self.params = params
        # TODO: há como tornar as ações contínuas? quais e quantos níveis de potência devem existir?
        self.actions = [i*params.p_max/10 for i in range(11)]
        self.states = [0,1]
        self.total_reward = 0
        self.reward = 0
        self.done = False        
        self.build_scenario()
    

    def build_scenario(self):
        # declaring the bs, mues and d2d pairs
        bs = base_station((0,0), radius = self.params.bs_radius)
        mues = [mobile_user(x) for x in range(self.params.n_mues)]
        d2d_pairs = [ (d2d_user(x, d2d_node_type.TX), d2d_user(x, d2d_node_type.RX)) for x in range(self.params.n_d2d) ]

        # distributing nodes in the bs radius
        gen.distribute_nodes(mues, bs)
        for p in d2d_pairs:
            gen.distribute_pair_fixed_distance( p, bs, self.params.d2d_pair_distance)

        # plot nodes positions
        # plot_positions(bs, mues, d2d_txs, d2d_rxs)

        # rb allocation
        # TODO: definir um metodo de alocacao de RBs. Nesta primeira simulacao, estou alocando todos para o mesmo RB. 
        # alocarei aleatoriamente nas proximas simulações
        for m in mues:
            m.set_rb(1)

        for p in d2d_pairs:
            p[0].set_rb(1)
            p[1].set_rb(1)

        # TODO: como determinar a potencia de transmissao do mue? vou setar pmax por enquanto
        for m in mues:
            m.set_tx_power(self.params.p_max)