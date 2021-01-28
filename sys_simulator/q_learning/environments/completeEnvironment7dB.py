from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from sys_simulator.channels import Channel
from sys_simulator.general import db_to_power, power_to_db
from sys_simulator.devices.devices \
    import base_station, mobile_user, d2d_user, d2d_node_type, node
from sys_simulator import general as gen
from sys_simulator.q_learning.agents.distanceAgent import DistanceAgent
from sys_simulator.q_learning.environments.environment import RLEnvironment
from typing import List
from sys_simulator.parameters.parameters import EnvironmentParameters
from scipy.spatial.distance import euclidean
from typing import Tuple
import numpy as np
import torch


class CompleteEnvironment7dB(RLEnvironment):
    """
    Same as CompleteEnvironment5dB, but made for convergence in one
    episode.
    """
    def __init__(self, params: EnvironmentParameters,
                 reward_function, channel: Channel, memory=1,
                 n_closest_devices=1, **kwargs):
        self.params = params
        super(CompleteEnvironment7dB, self).__init__(params,
                                                     reward_function, **kwargs)
        self.states = [0, 0, 1]
        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sinr_d2ds: float = []
        self.channel: Channel = channel
        self.pathlosses_are_calculated = False
        self.n_closest_devices = n_closest_devices
        self.memory = memory
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
        # set diff
        diff = self.n_closest_devices - len(self.d2d_pairs) + 1
        self.diff = 0 if diff < 0 else diff
        # dummy pairs
        for _ in range(self.diff):
            self.d2d_pairs.append(self.make_dummy_d2d_pair())
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
        # reset sets
        self.reset_sets()

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
        self.mue = mobile_user(0, self.params.p_max)
        self.mue.set_gain(self.params.user_gain)
        self.mue.set_position(mue_position)
        self.mue.set_rb(self.rb)
        self.mue.set_tx_power(self.params.p_max)
        self.mue.set_distance_to_bs(euclidean(mue_position, self.bs.position))
        # instantiate d2d_pairs
        self.d2d_pairs = [(d2d_user(x, d2d_node_type.TX, self.params.p_max),
                           d2d_user(x, d2d_node_type.RX, self.params.p_max))
                          for x in range(len(agents))]
        # set diff
        diff = self.n_closest_devices - len(self.d2d_pairs) + 1
        self.diff = 0 if diff < 0 else diff
        # dummy pairs
        for _ in range(self.diff):
            self.d2d_pairs.append(self.make_dummy_d2d_pair())
        self.rb = 1
        self.distances = [1/10*i*self.bs.radius for i in range(11)]
        # distributing nodes in the bs radius
        if euclidean(mue_position, self.bs.position) <= self.params.bs_radius:
            self.mue.set_position(mue_position)
        else:
            raise Exception(
                'Node distance to BS is greater than the BS radius.'
            )
        for pair, positions in zip(self.d2d_pairs, pairs_positions):
            # check if node is inside the BS radius
            if euclidean(
                positions[0], self.bs.position
            ) <= self.params.bs_radius:
                # set tx position
                pair[0].set_position(positions[0])
                pair[1].set_position(positions[1])
                # set tx distances
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
            # register d2d device to a RL agent
            agents[i].set_d2d_tx_id(self.d2d_pairs[i][0].id)
            agents[i].set_d2d_tx(self.d2d_pairs[i][0])
        # set diff: the amount of devices must be >= than
        # `self.n_closest_devices` in order to build the environment states
        diff = self.n_closest_devices - len(self.d2d_pairs) + 1
        self.diff = 0 if diff < 0 else diff
        # reset sets
        self.reset_sets()

    def get_state(self, agent: ExternalDQNAgent):
        # calculates all pathlosses
        if not self.pathlosses_are_calculated:
            # d2d pathlosses
            self.calculate_d2d_pathlosses()
            # mue pathloss
            self.calculate_mue_pathloss()
        d2d_tx: d2d_user = agent.d2d_tx
        close_pairs = self.get_n_closest_pairs(
            d2d_tx, self.n_closest_devices
        ) if len(self.d2d_pairs) > 1 else []
        close_txs, close_rxs = list(zip(*close_pairs))
        close_devices_x = [d[0].position[0] for d in close_pairs]
        close_devices_y = [d[0].position[1] for d in close_pairs]
        # last_mue_powers = self.mue.past_actions[:self.memory].tolist()
        # mue_sinrs = self.mue.past_sinrs[:self.memory].tolist()
        device_sinrs = d2d_tx.past_sinrs[:self.memory]
        device_powers = d2d_tx.past_actions[:self.memory].tolist()
        # d2d_pathloss = d2d_tx.pathloss_d2d
        close_devs_powers = []
        close_devs_sinrs = []
        close_interferences = []
        # close_inverse_link_prices = []
        close_speffs = []
        caused_interferences = []
        for d in close_txs:
            close_interferences.append(
                power_to_db(d2d_tx.get_received_interference(d.id))
            )
            caused_interferences.append(
                power_to_db(d.get_received_interference(d2d_tx.id))
            )
            # close_inverse_link_prices += d.avg_speffs[:self.memory].tolist()
            close_speffs += d.speffs[:self.memory].tolist()
            close_devs_powers += d.past_actions[:self.memory].tolist()
            close_devs_sinrs += d.past_sinrs[:self.memory].tolist()
        device_contrib = d2d_tx.past_actions[0] - d2d_tx.past_bs_losses[0]
        received_interference = d2d_tx.calc_received_interference()
        # mue states
        mue_speffs = self.mue.speffs[:self.memory]
        # mue_inverse_link_prices = self.mue.avg_speffs.tolist()
        # + d2d_tx.gain + self.bs.gain
        bs_interference = self.mue.past_actions[0] \
            - self.mue.past_bs_losses[0] - self.mue.past_sinrs[0]
        # + self.mue.gain + self.bs.gain
        device_contrib_pct = db_to_power(device_contrib - bs_interference)
        d2d_tx.set_interference_contrib_pct(device_contrib_pct)
        # recent_d2d_pathloss = d2d_tx.pathloss_d2d
        # recent_bs_pathloss = d2d_tx.pathloss_to_bs
        number_of_d2d_pairs = len(self.d2d_pairs)
        interference_indicator = self.mue.sinr > self.params.sinr_threshold
        # normalization
        device_sinrs = [np.clip(i, -30, 30) for i in device_sinrs]
        close_devs_sinrs = [np.clip(i, -30, 30) for i in close_devs_sinrs]
        # inverse_link_price = d2d_tx.avg_speffs[0]
        # state
        state = [
            number_of_d2d_pairs / 10,
            d2d_tx.position[0] / self.bs.radius,
            d2d_tx.position[1] / self.bs.radius,
            self.mue.position[0] / self.bs.radius,
            self.mue.position[1] / self.bs.radius,
            np.clip(agent.action, -30, 30) / 30,
            # inverse_link_price,
            # self.mue.tx_power / 30,
            int(interference_indicator),
            int(not interference_indicator),
            np.clip(received_interference, -30, 30) / 30,
        ]
        state += (np.array(close_devices_x) / self.bs.radius).tolist()
        state += (np.array(close_devices_y) / self.bs.radius).tolist()
        # state += (np.array(last_mue_powers) / 30).tolist()
        # state += (np.array(mue_sinrs) / 30).tolist()
        state += mue_speffs.tolist()
        # state += mue_inverse_link_prices
        state += (np.clip(device_sinrs, -30, 30) / 30).tolist()
        state += (np.clip(device_powers, -30, 30) / 30).tolist()
        state += (np.clip(close_interferences, -30, 30) / 30).tolist()
        # state.append(d2d_pathloss / 30)
        state += (np.clip(close_devs_powers, -30, 30) / 30).tolist()
        # state += (np.clip(close_devs_sinrs, -30, 30) / 30).tolist()
        # state += close_inverse_link_prices
        state += close_speffs
        state += (np.clip(caused_interferences, -30, 30) / 30).tolist()
        state += d2d_tx.speffs.tolist()
        state.append(np.clip(device_contrib, -30, 30) / 30)
        state.append(device_contrib_pct)
        # state.append(recent_d2d_pathloss / 30)
        # state.append(recent_bs_pathloss / 30)
        # state = db_to_power(torch.tensor(state)).view(1, -1).to(self.device)
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
        # calculate pathlosses
        if not self.pathlosses_are_calculated:
            # d2d pathlosses
            self.calculate_d2d_pathlosses()
            # mue pathloss
            self.calculate_mue_pathloss()
        # mue_tx_power
        mue_tx_power = self.mue.get_tx_power_db(
            self.bs, self.params.sinr_threshold, self.params.noise_power,
            self.params.mue_margin, self.params.p_max)
        self.mue.set_tx_power(mue_tx_power)
        # mue sinr
        _ = self.sinr_mue(
            self.mue, list(zip(*self.d2d_pairs))[0],
            self.params.noise_power,
            self.params.bs_gain, self.params.user_gain
        )
        self.mue.calc_set_avg_speff_set_link_price()
        self.mue.calc_set_speff()
        # d2d pairs sinr
        for p in self.d2d_pairs:
            if p[0].rb == self.rb:
                _ = self.sinr_d2d(
                    p[0], p[1], list(zip(*self.d2d_pairs))[0], self.mue,
                    self.params.noise_power, self.params.user_gain
                )
                p[0].calc_set_speff()
                p[0].calc_set_avg_speff_set_link_price()
        # get the states
        states = [self.get_state(a) for a in agents]
        # rewards
        rewards = [
            self.calculate_old_reward(a, self.params.c_param)
            for a in agents
        ]
        # total reward
        self.reward = np.sum(rewards)
        # spectral efficiencies
        self.mue_spectral_eff = self.mue.speffs[0]
        self.d2d_spectral_eff = np.sum(
            [d[0].speffs[0] for d in self.d2d_pairs]
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
            self.pathlosses_are_calculated = True

    def calculate_mue_pathloss(self):
        pathloss_to_bs = self.channel.step(
            euclidean(self.mue.position, self.bs.position)
        )
        self.mue.set_pathloss_to_bs(pathloss_to_bs)
        self.pathlosses_are_calculated = True

    def set_n_d2d(self, n_d2d):
        self.n_d2d = n_d2d

    def sinr_mue(self, mue: mobile_user, d2d_devices: List[d2d_user],
                 noise_power: float,
                 bs_gain: float, user_gain: float):
        mue_contrib = mue.tx_power + user_gain + bs_gain \
            - mue.pathloss_to_bs
        d2d_interferers = [d for d in d2d_devices if (
            d.type == d2d_node_type.TX and d.rb == mue.rb)]
        d2d_interferences = []
        interferences = []
        for d in d2d_interferers:
            interference = \
                db_to_power(
                    d.tx_power +
                    user_gain +
                    bs_gain -
                    d.pathloss_to_bs
                )
            d2d_interferences.append((d.id, interference))
            interferences.append(interference)
        mue.set_interferences(d2d_interferences)
        total_interference = np.sum(interferences)
        sinr = mue_contrib - power_to_db(
            db_to_power(noise_power) + total_interference
        )
        mue.set_sinr(sinr)
        return sinr

    def sinr_d2d(self, d2d_tx: d2d_user, d2d_rx: d2d_user,
                 d2d_devices: List[d2d_user], mue: mobile_user,
                 noise_power: float, user_gain: float):
        # if dummy device
        # if d2d_tx.is_dummy:
        # pass
        d2d_tx_contrib = d2d_tx.tx_power - \
            self.channel.step(d2d_tx.distance_d2d) + 2 * user_gain
        d2d_rx_mue_distance = euclidean(d2d_rx.position, mue.position)
        mue_interference = mue.tx_power - \
            self.channel.step(d2d_rx_mue_distance) + 2 * user_gain
        d2d_interferers = [
            d for d in d2d_devices if (
                d.id != d2d_tx.id
                and d.type == d2d_node_type.TX and d.rb == d2d_tx.rb
            )
        ]
        d2d_interferences = []
        for d in d2d_interferers:
            interference = \
                db_to_power(
                    d.tx_power +
                    2 * user_gain -
                    self.channel.step(
                        euclidean(d2d_rx.position, d.position)
                    )
                )
            d2d_interferences.append((d.id,  interference))
        d2d_tx.set_interferences(d2d_interferences)
        _, interferences = [i for i in zip(*d2d_interferences)]
        total_interference = np.sum(interferences)
        sinr = d2d_tx_contrib - \
            power_to_db(
                db_to_power(noise_power) +
                db_to_power(mue_interference) +
                total_interference
            )
        d2d_tx.set_sinr(sinr)
        return sinr

    def reset_sets(self):
        self.pathlosses_are_calculated = False
        self.mue.reset_set_flags()
        for d in self.d2d_pairs:
            d[0].reset_set_flags()

    def get_n_closest_pairs(
        self,
        device: d2d_user,
        n: int
    ) -> List[d2d_user]:
        pairs = [
            d for d in self.d2d_pairs if d[0].id != device.id
        ]
        distances = [
            euclidean(device.position, p[0].position) for p in pairs
        ]
        aux = [i for i in zip(pairs, distances)]
        aux.sort(key=lambda x: x[1])
        length = n if n <= len(pairs) else len(pairs)
        sorted_pairs, _ = zip(*aux)
        return list(sorted_pairs[:length])

    def get_state_size(self, foo_agent: ExternalDQNAgent):
        foo = self.get_state(foo_agent)
        return foo.shape[1]

    def rewards(self, agents: List[ExternalDQNAgent]):
        flag = self.mue.sinr >= self.params.sinr_threshold
        d2d_txs = [a.d2d_tx for a in agents]
        d2ds_speffs, d2d_powers, d2d_bs_pathlosses = zip(
            *[(d.speffs[0], d.tx_power, d.pathloss_to_bs) for d in d2d_txs]
        )
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
            if pct >= 0 and pct <= 1:
                if pct >= .05 and pct < .5:
                    beta = 50
                    alpha = .05
                elif pct >= .5 and pct < 1:
                    beta = 500
                    alpha = .005
            else:
                raise Exception('Invalid pct.')
        # with pathloss to bs
        reward = alpha * np.log2(1 + db_to_power(d2d_tx.sinr)) - beta * \
            np.log2(1 + db_to_power(d2d_tx.tx_power - d2d_tx.pathloss_to_bs))
        # without pathloss to bs
        # reward = alpha * np.log2(1 + db_to_power(d2d_tx.sinr)) - beta * \
        #     np.log2(1 + db_to_power(d2d_tx.tx_power))
        # dumb test reward
        # reward = alpha * np.log2(1 + db_to_power(d2d_tx.sinr))
        return reward

    def calculate_old_reward(
        self, agent: ExternalDQNAgent, C: int, penalty=1
    ) -> float:
        d2d_tx = agent.d2d_tx
        d2d_speff = d2d_tx.speffs[0]
        reward = -penalty
        if self.mue.sinr >= self.params.sinr_threshold:
            reward = 1/C * d2d_speff
        return reward

    def calculate_mue_speff_without_interferer(
        self,
        interferer: node,
    ):
        mue_contrib = \
            self.mue.tx_power + self.mue.gain + self.bs.gain \
            - self.mue.pathloss_to_bs
        mue_contrib = db_to_power(mue_contrib)
        interferences = [
            i[1] for i in self.mue.interferences if i[0] != interferer.id
        ]
        total_interference = np.sum(interferences)
        c = np.log2(
            1 + mue_contrib /
            (total_interference + db_to_power(self.params.noise_power))
        )
        return c

    def calculate_pi_mue(self, device: d2d_user):
        c = self.calculate_mue_speff_without_interferer(device)
        pi = self.mue.link_prices[0] * (c - self.mue.speffs[0])
        # if pi < 0:
        #     raise Exception('pi cannot be negative.')
        return pi

    def calculate_reward_art(self, agent: ExternalDQNAgent):
        device = agent.d2d_tx
        pi = self.calculate_pi_mue(device)
        reward = \
            device.link_prices[0] * device.speffs[0] - pi
        if reward > 1e3:
            print('bug')
        return reward

    def make_dummy_d2d_pair(self) -> Tuple[d2d_user, d2d_user]:
        device = d2d_user(
            len(self.d2d_pairs),
            d2d_node_type.TX,
            self.params.p_max
        )
        device.set_distance_to_bs(self.bs.radius)
        device.set_distance_to_mue(self.bs.radius)
        device.set_position((self.bs.radius, 0))
        device.set_rb(1)
        memory_size = self.memory
        device.past_sinrs = 30 * np.ones(memory_size)
        device.past_d2d_losses = np.ones(memory_size)
        device.set_tx_power(-1000)
        device.set_pathloss_to_bs(1000)
        device.set_distance_d2d(self.params.d2d_pair_distance)
        device.speffs = 30 * np.ones(memory_size)
        device.avg_speffs = 30 * np.ones(memory_size)
        device.link_prices = 1/30 * np.ones(memory_size)
        device.is_dummy = True
        device_rx = d2d_user(len(self.d2d_pairs), d2d_node_type.RX,
                             self.params.p_max)
        device_rx.set_position(
            (self.bs.radius - self.params.d2d_pair_distance, 0)
        )
        device_rx.set_rb(1)
        device_rx.is_dummy = True
        return (device, device_rx)
