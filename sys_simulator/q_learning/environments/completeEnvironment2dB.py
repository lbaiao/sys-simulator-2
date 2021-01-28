from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from sys_simulator.q_learning.agents.agent import Agent
from sys_simulator.devices.devices \
    import base_station, mobile_user, d2d_user, d2d_node_type
from sys_simulator import general as gen
from sys_simulator.sinr.sinr import sinr_d2d_db, sinr_mue_db
from sys_simulator.q_learning.agents.distanceAgent import DistanceAgent
from sys_simulator.q_learning.environments.environment import RLEnvironment
from typing import List, Tuple
from sys_simulator.parameters.parameters import EnvironmentParameters
from scipy.spatial.distance import euclidean
import numpy as np
import torch


class CompleteEnvironment2dB(RLEnvironment):
    """
    Same as CompleteEnvironment, but everything is dB
    """
    def __init__(self, params: EnvironmentParameters,
                 reward_function, **kwargs):
        self.params = params
        # TODO: há como tornar as ações contínuas? quais e quantos
        # níveis de potência devem existir?
        # self.actions = [i*params.p_max/10 for i in range(11)]
        super(CompleteEnvironment2dB, self).__init__(params,
                                                     reward_function, **kwargs)
        self.states = [0, 0, 1]
        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sinr_d2ds = []

    def build_scenario(self, agents: List[DistanceAgent]):
        # declaring the bs, mues and d2d pairs
        self.sinr_d2ds = []
        self.bs = base_station((0, 0),
                               radius=self.params.bs_radius)
        self.bs.set_gain(self.params.bs_gain)
        self.mue = mobile_user(0)
        self.mue.set_gain(self.params.user_gain)
        self.d2d_pairs = [(d2d_user(x, d2d_node_type.TX),
                           d2d_user(x, d2d_node_type.RX))
                          for x in range(len(agents))]
        self.rb = 1
        # distributing nodes in the bs radius
        gen.distribute_nodes([self.mue], self.bs)
        for p in self.d2d_pairs:
            gen.distribute_pair_fixed_distance(
                p, self.bs, self.params.d2d_pair_distance
            )
            for d in p:
                d.set_distance_d2d(self.params.d2d_pair_distance)
                d.set_gain(self.params.user_gain)

        self.mue.set_rb(self.rb)

        for p in self.d2d_pairs:
            p[0].set_rb(self.rb)
            p[1].set_rb(self.rb)

        # TODO: como determinar a potencia de transmissao do mue?
        # vou setar pmax por enquanto
        self.mue.set_tx_power(self.params.p_max)

        for i in range(len(agents)):
            agents[i].set_d2d_tx_id(self.d2d_pairs[i][0].id)

    def set_scenario(self, pairs_positions: List[Tuple],
                     mue_position: Tuple, agents: List[Agent]):
        if len(pairs_positions) != len(agents):
            raise Exception('Different `pair_positions` and `agents` lengths.')
        # declaring the bs, mues and d2d pairs
        self.sinr_d2ds = []
        self.rb = 1
        self.bs = base_station((0, 0), radius=self.params.bs_radius)
        self.bs.set_gain(self.params.bs_gain)
        # mue stuff
        self.mue = mobile_user(0)
        self.mue.set_gain(self.params.user_gain)
        self.mue.set_position(mue_position)
        self.mue.set_rb(self.rb)
        self.mue.set_tx_power(self.params.p_max)
        self.mue.set_distance_to_bs(euclidean(mue_position, self.bs.position))
        # d2d_pairs
        self.d2d_pairs = [(d2d_user(x, d2d_node_type.TX),
                           d2d_user(x, d2d_node_type.RX))
                          for x in range(len(agents))]
        # distributing nodes in the bs radius
        if euclidean(mue_position, self.bs.position) <= self.params.bs_radius:
            self.mue.set_position(mue_position)
        else:
            raise Exception(
                'Node distance to BS is greater than the BS radius.'
            )
        for pair, position in zip(self.d2d_pairs, pairs_positions):
            if euclidean(position, self.bs.position) <= self.params.bs_radius:
                pair[0].set_position(position)
                gen.distribute_rx_fixed_distance(
                    pair, self.bs, self.params.d2d_pair_distance
                )
                for d in pair:
                    d.set_distance_d2d(self.params.d2d_pair_distance)
                    d.set_distance_to_bs(euclidean(d.position,
                                                   self.bs.position))
                    d.set_gain(self.params.user_gain)
                    d.set_rb(self.rb)
            else:
                raise Exception(
                    'Node distance to BS is greater than the BS radius.'
                )
        for i in range(len(agents)):
            agents[i].set_d2d_tx_id(self.d2d_pairs[i][0].id)

    def get_state(self, agent: ExternalDQNAgent):
        sinr = self.mue.sinr
        (index, d2d_tx) =\
            [(index, p[0]) for index, p
                in enumerate(self.d2d_pairs) if p[0].id == agent.id][0]
        d2d_rx = self.d2d_pairs[index][1]

        number_of_d2d_pairs = len(self.d2d_pairs)
        d2d_tx_distance_to_bs = d2d_tx.distance_to_bs
        d2d_rx_distance_to_mue = euclidean(d2d_rx.position, self.mue.position)
        mue_distance_to_bs = self.mue.distance_to_bs

        interference_indicator = sinr > self.params.sinr_threshold

        # normalization
        d2d_tx_distance_to_bs /= self.params.bs_radius
        d2d_rx_distance_to_mue /= 2*self.params.bs_radius
        mue_distance_to_bs /= self.params.bs_radius

        state = torch.tensor(
            [[
                number_of_d2d_pairs,
                d2d_tx_distance_to_bs,
                d2d_rx_distance_to_mue,
                mue_distance_to_bs,
                # agent.action,
                # self.mue.tx_power,
                int(interference_indicator),
                int(not interference_indicator)
            ]]).to(self.device)

        return state

    def step(self, agents: List[DistanceAgent]):
        # allocate agents tx power
        for agent in agents:
            for pair in self.d2d_pairs:
                if agent.id == pair[0].id:
                    pair[0].tx_power = agent.action
        # mue_tx_power
        mue_tx_power = self.mue.get_tx_power_db(
            self.bs, self.params.sinr_threshold, self.params.noise_power,
            self.params.mue_margin, self.params.p_max)
        self.mue.set_tx_power(mue_tx_power)
        # mue sinr
        sinr_m = sinr_mue_db(self.mue, list(zip(*self.d2d_pairs))[0],
                             self.bs, self.params.noise_power,
                             self.params.bs_gain, self.params.user_gain)
        self.mue.set_sinr(sinr_m)
        # d2d pairs sinr
        sinr_d2ds = list()
        for p in self.d2d_pairs:
            if p[0].rb == self.rb:
                sinr_d = sinr_d2d_db(
                    p[0], p[1], list(zip(*self.d2d_pairs))[0], self.mue,
                    self.params.noise_power, self.params.user_gain
                )
                sinr_d2ds.append(sinr_d)
        self.sinr_d2ds.append(sinr_d2ds)
        sinr_d2ds = np.array(sinr_d2ds)
        # get the states
        states = [self.get_state(a) for a in agents]
        flag = True
        if sinr_m < self.params.sinr_threshold:
            flag = False
        # rewards
        rewards, mue_se, d2d_se = self.reward_function(
            sinr_m, sinr_d2ds, flag, self.params.c_param, penalty=5
        )
        # early stopping
        done = not flag
        # total reward
        self.reward = torch.sum(rewards)
        # spectral efficiencies
        self.mue_spectral_eff = mue_se
        self.d2d_spectral_eff = d2d_se
        # end
        return states, rewards, done
