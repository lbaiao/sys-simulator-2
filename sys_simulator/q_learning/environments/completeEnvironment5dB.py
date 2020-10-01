from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from sys_simulator.channels import Channel
from sys_simulator.general.general import db_to_power, power_to_db
from sys_simulator.devices.devices \
    import base_station, mobile_user, d2d_user, d2d_node_type
from sys_simulator.general import general as gen
from sys_simulator.q_learning.agents.distanceAgent import DistanceAgent
from sys_simulator.q_learning.environments.environment import RLEnvironment
from typing import List
from sys_simulator.parameters.parameters import EnvironmentParameters
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
        self.sinr_d2ds: float = []
        self.channel: Channel = channel
        self.pathlosses_are_calculated = False
        self.n_closest_devices = 2
        self.memory = 2
        self.dummy_d2d_pair = (
            self.make_dummy_d2d_device(d2d_node_type.TX),
            self.make_dummy_d2d_device(d2d_node_type.RX)
        )
        self.diff = 0

    def build_scenario(self, agents: List[ExternalDQNAgent]):
        # declaring the bs, mues and d2d pairs
        self.sinr_d2ds = []
        self.bs = base_station((0, 0),
                               radius=self.params.bs_radius)
        self.bs.set_gain(self.params.bs_gain)
        self.mue = mobile_user(0, self.params.p_max)
        self.mue.set_gain(self.params.user_gain)
        self.d2d_pairs = [(d2d_user(x, d2d_node_type.TX, self.params.p_max),
                           d2d_user(x, d2d_node_type.RX, self.params.p_max))
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
            agents[i].set_d2d_tx(self.d2d_pairs[i][0])
        # set diff
        diff = self.n_closest_devices - len(self.d2d_pairs) + 1
        self.diff = 0 if diff < 0 else diff

    def set_scenario(self, pairs_positions: List[Tuple],
                     mue_position: Tuple, agents: List[ExternalDQNAgent]):
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
        self.d2d_pairs = [(d2d_user(x, d2d_node_type.TX, self.params.p_max),
                           d2d_user(x, d2d_node_type.RX, self.params.p_max))
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
            agents[i].set_d2d_tx(self.d2d_pairs[i][0])
        # set diff
        diff = self.n_closest_devices - len(self.d2d_pairs) + 1
        self.diff = 0 if diff < 0 else diff

    def get_state(self, agent: ExternalDQNAgent):
        # calculates all pathlosses
        if not self.pathlosses_are_calculated:
            # d2d pathlosses
            self.calculate_d2d_pathlosses()
            # mue pathloss
            self.calculate_mue_pathloss()
        sinr = self.sinr_mue(
            self.mue, list(zip(*self.d2d_pairs))[0],
            self.params.noise_power,
            self.params.bs_gain, self.params.user_gain
        )
        d2d_tx = agent.d2d_tx
        close_devices = self.get_n_closest_transmitters(
            d2d_tx, self.n_closest_devices
        ) if len(self.d2d_pairs) > 1 else []
        for _ in range(self.diff):
            # append a dummy pair
            close_devices.append(self.dummy_d2d_pair[0])
        close_devices_x = [d.position[0] for d in close_devices]
        close_devices_y = [d.position[1] for d in close_devices]
        last_mue_powers = self.mue.past_actions[:self.memory].tolist()
        mue_sinrs = self.mue.past_sinrs[:self.memory].tolist()
        device_sinrs = d2d_tx.past_sinrs[:self.memory]
        device_powers = d2d_tx.past_actions[:self.memory].tolist()
        d2d_pathloss = d2d_tx.pathloss_d2d
        close_devs_powers = []
        close_devs_sinrs = []
        for d in close_devices:
            close_devs_powers += d.past_actions[:self.memory].tolist()
            close_devs_sinrs += d.past_sinrs[:self.memory].tolist()
        device_contrib = d2d_tx.past_actions[0] - d2d_tx.past_bs_losses[0]
        # + d2d_tx.gain + self.bs.gain
        bs_interference = self.mue.past_actions[0] \
            - self.mue.past_bs_losses[0] - self.mue.past_sinrs[0]
        # + self.mue.gain + self.bs.gain
        device_contrib_pct = db_to_power(device_contrib - bs_interference)
        d2d_tx.set_interference_contrib_pct(device_contrib_pct)
        recent_d2d_pathloss = d2d_tx.pathloss_d2d
        recent_bs_pathloss = d2d_tx.pathloss_to_bs
        number_of_d2d_pairs = len(self.d2d_pairs)
        interference_indicator = sinr > self.params.sinr_threshold
        # normalization
        # d2d_tx_distance_to_bs /= self.params.bs_radius
        # d2d_rx_distance_to_mue /= 2*self.params.bs_radius
        # mue_distance_to_bs /= self.params.bs_radius
        device_sinrs = [gen.ceil(i, 30) for i in device_sinrs]
        close_devs_sinrs = [gen.ceil(i, 30) for i in close_devs_sinrs]
        # state
        state = [
            number_of_d2d_pairs,
            d2d_tx.position[0],
            d2d_tx.position[1],
            self.mue.position[0],
            self.mue.position[1],
            agent.action,
            self.mue.tx_power,
            int(interference_indicator),
            int(not interference_indicator),
        ]
        state += close_devices_x
        state += close_devices_y
        state += last_mue_powers
        state += mue_sinrs
        state += device_sinrs
        state += device_powers
        state.append(d2d_pathloss)
        state += close_devs_powers
        state += close_devs_sinrs
        state.append(device_contrib)
        state.append(device_contrib_pct)
        state.append(recent_d2d_pathloss)
        state.append(recent_bs_pathloss)
        state = torch.tensor([state]).to(self.device)
        # end
        self.reset_sets()
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
                    pair[0].set_tx_power(agent.action)
        # mue_tx_power
        mue_tx_power = self.mue.get_tx_power_db(
            self.bs, self.params.sinr_threshold, self.params.noise_power,
            self.params.mue_margin, self.params.p_max)
        self.mue.set_tx_power(mue_tx_power)
        # mue sinr
        sinr_m = self.sinr_mue(
            self.mue, list(zip(*self.d2d_pairs))[0],
            self.params.noise_power,
            self.params.bs_gain, self.params.user_gain
        )
        self.mue.set_sinr(sinr_m)
        # d2d pairs sinr
        for p in self.d2d_pairs:
            if p[0].rb == self.rb:
                _ = self.sinr_d2d(
                    p[0], p[1], list(zip(*self.d2d_pairs))[0], self.mue,
                    self.params.noise_power, self.params.user_gain
                )
        # get the states
        states = [self.get_state(a) for a in agents]
        # rewards
        rewards = [self.calculate_reward(a) for a in agents]
        # total reward
        self.reward = np.sum(rewards)
        # spectral efficiencies
        self.mue_spectral_eff = np.log2(1 + db_to_power(self.mue.sinr))
        self.d2d_spectral_eff = np.sum(
            [np.log2(1 + db_to_power(d[0].sinr)) for d in self.d2d_pairs]
        )
        # end
        return states, rewards

    def calculate_d2d_pathlosses(self):
        for tx in [d[0] for d in self.d2d_pairs]:
            pathloss_to_bs = self.channel.step(
                euclidean(tx.position, self.bs.position)
            )
            pathloss_to_rx = self.channel.step(self.params.d2d_pair_distance)
            tx.set_pathloss_to_bs(pathloss_to_bs)
            tx.set_pathloss_d2d(pathloss_to_rx)

    def calculate_mue_pathloss(self):
        pathloss_to_bs = self.channel.step(
            euclidean(self.mue.position, self.bs.position)
        )
        self.mue.set_pathloss_to_bs(pathloss_to_bs)

    def set_n_d2d(self, n_d2d):
        self.n_d2d = n_d2d

    def sinr_mue(self, mue: mobile_user, d2d_devices: List[d2d_user],
                 noise_power: float,
                 bs_gain: float, user_gain: float):
        mue_contrib = mue.tx_power + user_gain + bs_gain \
            - mue.pathloss_to_bs
        d2d_interferers = [d for d in d2d_devices if (
            d.type == d2d_node_type.TX and d.rb == mue.rb)]
        d2d_interference = sum(
            [
                db_to_power(d.tx_power) *
                db_to_power(user_gain) *
                db_to_power(bs_gain) /
                db_to_power(d.pathloss_to_bs)
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
        d2d_tx.set_sinr(sinr)
        return sinr

    def reset_sets(self):
        self.pathlosses_are_calculated = False
        self.mue.reset_set_flags()
        for d in self.d2d_pairs:
            d[0].reset_set_flags()

    def get_n_closest_transmitters(
        self,
        device: d2d_user,
        n: int
    ) -> List[d2d_user]:
        transmitters = [
            d[0] for d in self.d2d_pairs if d[0].id != device.id
        ]
        distances = [
            euclidean(device.position, t.position) for t in transmitters
        ]
        aux = [i for i in zip(transmitters, distances)]
        aux.sort(key=lambda x: x[1])
        length = n if n <= len(transmitters) else len(transmitters)
        sorted_txs, _ = zip(*aux)
        return list(sorted_txs[:length])

    def get_state_size(self, foo_agent: ExternalDQNAgent):
        foo = self.get_state(foo_agent)
        return foo.shape[1]

    def rewards(self, agents: List[ExternalDQNAgent]):
        flag = self.mue.sinr >= self.params.sinr_threshold
        d2d_txs = [a.d2d_tx for a in agents]
        d2d_sinrs, d2d_powers, d2d_bs_pathlosses = zip(
            *[(d.sinr, d.tx_power, d.pathloss_to_bs) for d in d2d_txs]
        )
        d2ds_speffs = np.log2(1 + db_to_power(d2d_sinrs))
        d2d_interferences = \
            np.log2(1 + db_to_power(d2d_powers - d2d_bs_pathlosses))
        beta = 10
        if flag:
            beta = 1
        rewards = d2ds_speffs - beta * d2d_interferences
        return rewards

    def calculate_reward(self, agent: ExternalDQNAgent) -> float:
        flag = self.mue.sinr < self.params.sinr_threshold
        pct = agent.d2d_tx.interference_contrib_pct
        d2d_tx = agent.d2d_tx
        beta = 1
        alpha = 1
        if flag:
            if pct >= 0 and pct < .05:
                beta = 1
                alpha = 1
            elif pct >= .05 and pct < .5:
                beta = 100
                alpha = .1
            elif pct >= .5 and pct < 1:
                beta = 1000
                alpha = .01
            else:
                raise Exception('Invalid pct.')
        # with pathloss to bs
        # reward = alpha * np.log2(1 + db_to_power(d2d_tx.sinr)) - beta * \
        #     np.log2(1 + db_to_power(d2d_tx.tx_power - d2d_tx.pathloss_to_bs))
        # without pathloss to bs
        reward = alpha * np.log2(1 + db_to_power(d2d_tx.sinr)) - beta * \
            np.log2(1 + db_to_power(d2d_tx.tx_power))
        return reward

    def make_dummy_d2d_device(self, d2d_type: d2d_node_type) -> d2d_user:
        device = d2d_user(99, d2d_type)
        device.set_distance_to_bs(1000)
        device.set_distance_to_mue(1000)
        device.set_position((1000, 0))
        device.set_rb(1)
        device.set_sinr(30)
        device.set_tx_power(-1000)
        device.set_pathloss_to_bs(1000)
        device.set_distance_d2d(0)
        return device
