from sys_simulator.channels import Channel
from sys_simulator.general.general import db_to_power, power_to_db
from sys_simulator.devices.devices \
    import base_station, mobile_user, d2d_user, d2d_node_type
from sys_simulator.general import general as gen
from sys_simulator.q_learning.agents.distanceAgent import DistanceAgent
from sys_simulator.q_learning.environments.environment import RLEnvironment
from typing import List
from sys_simulator.parameters.parameters import EnvironmentParameters
from sys_simulator.q_learning.agents.agent import Agent
from scipy.spatial.distance import euclidean
from typing import Tuple
import numpy as np
import torch


class CompleteEnvironment5dB(RLEnvironment):
    """
    Same as CompleteEnvironment5, but everything is in dB.
    """
    def __init__(self, params: EnvironmentParameters,
                 reward_function, channel: Channel, **kwargs):
        self.params = params
        super(CompleteEnvironment5dB, self).__init__(params,
                                                     reward_function, **kwargs)
        self.states = [0, 0, 1]
        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sinr_d2ds = []
        self.channel = channel
        self.obs_size = 13

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
        self.distances = [1/10*i*self.bs.radius for i in range(11)]
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

    def get_state(self, agent: DistanceAgent):
        sinr = self.sinr_mue(
            self.mue, list(zip(*self.d2d_pairs))[0],
            self.bs, self.params.noise_power,
            self.params.bs_gain, self.params.user_gain
        )
        (index, d2d_tx) =\
            [(index, p[0]) for index, p
                in enumerate(self.d2d_pairs) if p[0].id == agent.id][0]
        # d2d_rx = self.d2d_pairs[index][1]

        number_of_d2d_pairs = len(self.d2d_pairs)
        # d2d_tx_distance_to_bs = d2d_tx.distance_to_bs
        # d2d_rx_distance_to_mue = \
        #   euclidean(d2d_rx.position, self.mue.position)
        # mue_distance_to_bs = self.mue.distance_to_bs

        interference_indicator = sinr > self.params.sinr_threshold
        x_mean, y_mean = self.get_other_devices_mean_positions(d2d_tx)
        x_std, y_std = self.get_other_devices_std_positions(d2d_tx)
        # normalization
        # d2d_tx_distance_to_bs /= self.params.bs_radius
        # d2d_rx_distance_to_mue /= 2*self.params.bs_radius
        # mue_distance_to_bs /= self.params.bs_radius

        state = torch.tensor(
            [[
                number_of_d2d_pairs,
                d2d_tx.position[0],
                d2d_tx.position[1],
                self.mue.position[0],
                self.mue.position[1],
                x_mean,
                y_mean,
                x_std,
                y_std,
                agent.action,
                self.mue.tx_power,
                int(interference_indicator),
                int(not interference_indicator),
            ]]).to(self.device)

        return state

    def get_other_devices_mean_positions(self, tx: d2d_user):
        other_devices = [d[0] for d in self.d2d_pairs if d[0] != tx]
        x_mean = np.mean([d.position[0] for d in other_devices])
        y_mean = np.mean([d.position[1] for d in other_devices])
        return x_mean, y_mean

    def get_other_devices_std_positions(self, tx: d2d_user):
        other_devices = [d[0] for d in self.d2d_pairs if d[0] != tx]
        x_std = np.std([d.position[0] for d in other_devices])
        y_std = np.std([d.position[1] for d in other_devices])
        return x_std, y_std

    def step(self, agents: List[DistanceAgent]):
        # allocate agents tx power
        for agent in agents:
            for pair in self.d2d_pairs:
                if agent.id == pair[0].id:
                    pair[0].tx_power = agent.action
        # mue_tx_power
        mue_tx_power = self.mue.get_tx_power_db(
            self.bs, self.params.sinr_threshold, self.params.noise_power,
            self.params.mue_margin, self.params.p_max, self.channel)
        self.mue.set_tx_power(mue_tx_power)
        # mue sinr
        sinr_m = self.sinr_mue(
            self.mue, list(zip(*self.d2d_pairs))[0],
            self.bs, self.params.noise_power,
            self.params.bs_gain, self.params.user_gain
        )
        self.mue.set_sinr(sinr_m)
        # d2d pairs sinr
        sinr_d2ds = list()
        for p in self.d2d_pairs:
            if p[0].rb == self.rb:
                sinr_d = self.sinr_d2d(
                    p[0], p[1], list(zip(*self.d2d_pairs))[0], self.mue,
                    self.params.noise_power, self.params.user_gain
                )
                sinr_d2ds.append(sinr_d)
        self.sinr_d2ds.append(sinr_d2ds)
        sinr_d2ds = np.array(sinr_d2ds)
        # get the states
        states = [self.get_state(a) for a in agents]
        flag = sinr_m >= self.params.sinr_threshold
        # rewards
        rewards, mue_se, d2d_se = self.reward_function(
            sinr_m, sinr_d2ds.tolist(), flag, self.params.c_param, penalty=5
        )
        done = not flag
        # total reward
        self.reward = torch.sum(rewards)
        # spectral efficiencies
        self.mue_spectral_eff = mue_se
        self.d2d_spectral_eff = d2d_se
        # end
        return states, rewards, done

    def set_n_d2d(self, n_d2d):
        self.n_d2d = n_d2d

    def sinr_mue(self, mue: mobile_user, d2d_devices: List[d2d_user],
                 bs: base_station, noise_power: float,
                 bs_gain: float, user_gain: float):
        mue_contrib = mue.tx_power + user_gain + bs_gain \
            - self.channel.step(mue.distance_to_bs)
        d2d_interferers = [d for d in d2d_devices if (
            d.type == d2d_node_type.TX and d.rb == mue.rb)]
        d2d_interference = sum(
            [
                db_to_power(d.tx_power) *
                db_to_power(user_gain) *
                db_to_power(bs_gain) /
                db_to_power(
                    self.channel.step(euclidean(d.position, bs.position))
                )
                for d in d2d_interferers
            ]
        )
        sinr = mue_contrib - power_to_db(
            db_to_power(noise_power) + d2d_interference
        )
        return sinr

    def sinr_d2d(self, d2d_tx: d2d_user, d2d_rx: d2d_user,
                 d2d_devices: List[d2d_user], mue: mobile_user,
                 noise_power: float, user_gain: float):
        d2d_tx_contrib = d2d_tx.tx_power - \
            self.channel.step(d2d_tx.distance_d2d) + 2 * user_gain
        d2d_rx_mue_distance = euclidean(d2d_rx.position, mue.position)
        mue_interference = mue.tx_power - \
            self.channel.step(d2d_rx_mue_distance) + 2 * user_gain
        d2d_interferers = [d for d in d2d_devices if (
            d.id != d2d_tx.id
            and d.type == d2d_node_type.TX and d.rb == d2d_tx.rb
        )]
        d2d_interference = sum(
            [
                db_to_power(d.tx_power) *
                db_to_power(2 * user_gain) /
                db_to_power(self.channel.step(
                    euclidean(d2d_rx.position, d.position)
                ))
                for d in d2d_interferers
            ]
        )
        sinr = d2d_tx_contrib - \
            power_to_db(
                db_to_power(noise_power) +
                db_to_power(mue_interference) +
                d2d_interference
            )
        return sinr
