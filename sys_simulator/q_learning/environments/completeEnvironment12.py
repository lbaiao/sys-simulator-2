from types import MethodType
from typing import List, Tuple
from sys_simulator.general.simple_memory import SimpleMemory

import numpy as np
from scipy.spatial.distance import euclidean

from sys_simulator import general as gen
from sys_simulator.channels import Channel
from sys_simulator.ddpg.agent import Agent, SurrogateAgent
from sys_simulator.devices.devices import (
    base_station,
    d2d_node_type,
    d2d_user,
    mobile_user,
    node,
)
from sys_simulator.dqn.agents.dqnAgent import ExternalDQNAgent
from sys_simulator.general import db_to_power, jain_index, power_to_db
from sys_simulator.parameters.parameters import EnvironmentParameters
from sys_simulator.q_learning.agents.distanceAgent import DistanceAgent
from sys_simulator.q_learning.environments.environment import RLEnvironment


class CompleteEnvironment12(RLEnvironment):
    """
    Similar to CompleteEnvironment9dB, but only the small scale fading changes
    if the device stays at the same position.
    It has different channels to the BS and to the devices.
    The channel is dynamic, and the devices have channel information,
    just before the transmission.

    Parameters
    ----
    reward_function: str
        options: 'continuous', 'sinrs', 'jain'. 
    states_options: List[str]
        options: 'sinrs', 'positions', 'channels'.
    """
    d2d_pairs: List[Tuple[d2d_user, d2d_user]]

    def __init__(
        self,
        params: EnvironmentParameters,
        channel_to_bs: Channel,
        channel_to_devices: Channel,
        memory=1,
        n_closest_devices=1,
        reward_penalty=1,
        bs_height=25,
        states_options = ['sinrs', 'positions'],
        reward_function='continuous',
        memories_capacity=int(1e3),
        **kwargs
    ):
        self.params = params
        super(CompleteEnvironment12, self).__init__(params,
                                                    reward_function,
                                                    **kwargs)
        self.states = [0, 0, 1]
        self.states_options = states_options
        self.mue: mobile_user = None
        self.sinr_d2ds: float = []
        self.d2d_pairs: List[Tuple[d2d_user, d2d_user]] = []
        self.channel_to_bs: Channel = channel_to_bs
        self.channel_to_devices: Channel = channel_to_devices
        self.n_closest_devices = n_closest_devices
        self.memory = memory
        self.diff = 0
        self.reward_penalty = reward_penalty
        self.bs_height = bs_height
        # memories
        self.powers_memory = SimpleMemory(memories_capacity)
        self.interferences_memory = SimpleMemory(memories_capacity)
        self.losses_to_bs_memory = SimpleMemory(memories_capacity)
        self.losses_to_d2d_memory = SimpleMemory(memories_capacity)
        self.rewards_memory = SimpleMemory(memories_capacity)
        self.sinrs_memory = SimpleMemory(memories_capacity)
        # losses
        self.pathlosses = {}
        self.small_scale_fadings = {}
        self.large_scale_fadings = {}
        self.total_losses = {}
        # flags
        self.pathlosses_are_set = False
        self.large_scale_fadings_are_set = False
        self.small_scale_fadings_are_set = False
        self.total_losses_are_set = False
        self.pathlosses_are_calculated = False
        # reward_function
        self.manage_reward_function(reward_function)

    def manage_reward_function(self, reward_function: str):
        # reward function
        if reward_function == 'sinr':
            self.reward_function = self.calculate_sinr_reward
        elif reward_function == 'continuous':
            self.reward_function = self.calculate_continuous_reward
        elif reward_function == 'jain':
            self.reward_function = self.calculate_jain_reward
        else:
            raise Exception('Invalid reward function.')


    def build_scenario(self, agents: List[SurrogateAgent], d2d_limited_power=True):
        # declaring the bs, mues and d2d pairs
        self.reset_before_build_set()
        self.sinr_d2ds = []
        self.bs = base_station((0, 0, self.bs_height),
                               radius=self.params.bs_radius)
        self.bs.set_gain(self.params.bs_gain)
        self.mue = mobile_user(0, self.params.p_max)
        self.mue.set_gain(self.params.user_gain)
        self.d2d_pairs = [
            (d2d_user(x, d2d_node_type.TX, self.params.p_max,
                      memory_size=self.memory, limited_power=d2d_limited_power),
             d2d_user(x, d2d_node_type.RX, self.params.p_max,
                      memory_size=self.memory, limited_power=d2d_limited_power))
            for x in range(len(agents))
        ]
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
        # pathlosses
        self.calc_set_all_losses()
        # reset sets
        self.reset_sets()

    def set_scenario(self, pairs_positions: List[Tuple],
                     mue_position: Tuple, agents: List[SurrogateAgent]):
        if len(pairs_positions) != len(agents):
            raise Exception('Different `pair_positions` and `agents` lengths.')
        self.reset_before_build_set()
        # declaring the bs, mues and d2d pairs
        self.sinr_d2ds = []
        self.rb = 1
        self.bs = base_station((0, 0, self.bs_height),
                               radius=self.params.bs_radius)
        self.bs.set_gain(self.params.bs_gain)
        # mue stuff
        self.mue = mobile_user(0, self.params.p_max)
        self.mue.set_gain(self.params.user_gain)
        self.mue.set_position(mue_position)
        self.mue.set_rb(self.rb)
        self.mue.set_tx_power(self.params.p_max)
        self.mue.set_distance_to_bs(euclidean(mue_position, self.bs.position))
        # instantiate d2d_pairs
        self.d2d_pairs = [(
            d2d_user(x, d2d_node_type.TX, self.params.p_max,
                     memory_size=self.memory),
            d2d_user(x, d2d_node_type.RX, self.params.p_max,
                     memory_size=self.memory)
        ) for x in range(len(agents))]
        # set diff
        diff = self.n_closest_devices - len(self.d2d_pairs) + 1
        self.diff = 0 if diff < 0 else diff
        # dummy pairs
        for _ in range(self.diff):
            self.d2d_pairs.append(self.make_dummy_d2d_pair())
        self.rb = 1
        self.distances = [1/10*i*self.bs.radius for i in range(11)]
        # distributing nodes in the bs radius
        if euclidean(mue_position[:2], self.bs.position[:2]) \
                <= self.params.bs_radius:
            self.mue.set_position(mue_position)
        else:
            raise Exception(
                'Node distance to BS is greater than the BS radius.'
            )
        for pair, positions in zip(self.d2d_pairs, pairs_positions):
            # check if node is inside the BS radius
            if euclidean(
                positions[0][:2], self.bs.position[:2]
            ) <= self.params.bs_radius and not pair[0].is_dummy:
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
        # pathlosses
        self.calc_set_all_losses()
        # reset sets
        self.reset_sets()

    def get_state(self):
        # get states
        positions = []
        sinrs = []
        losses_to_bs = []
        losses_to_d2d = []
        for tx, rx in self.d2d_pairs:
            positions += tx.position[:2]
            positions += rx.position[:2]
            sinrs.append(tx.sinr)
            loss_bs = self.total_losses[tx.id][self.bs.id]
            loss_d2d = self.total_losses[tx.id][rx.id]
            losses_to_bs.append(loss_bs)
            losses_to_d2d.append(loss_d2d)
        sinrs.append(self.mue.sinr)
        positions += self.mue.position[:2]
        losses_to_bs.append(self.total_losses[self.mue.id][self.bs.id])
        # states
        states = []
        # normalize and add positions
        if 'positions' in self.states_options:
            positions = np.array(positions)
            positions = self.norm_position(positions)
            states.append(positions)
        # normalize and add sinrs
        if 'sinrs' in self.states_options:
            sinrs = np.array(sinrs)
            sinrs = self.normalize(
                sinrs, self.sinrs_memory, self.min_max_scaling)
            states.append(sinrs)
        # normalize and add channel losses
        if 'channels' in self.states_options:
            losses_to_bs = np.array(losses_to_bs)
            losses_to_d2d = np.array(losses_to_d2d)
            losses_to_bs = self.normalize(
                losses_to_bs, self.losses_to_bs_memory, self.min_max_scaling)
            losses_to_d2d = self.normalize(
                losses_to_d2d, self.losses_to_d2d_memory, self.min_max_scaling)
            states.append(losses_to_bs)
            states.append(losses_to_d2d)
        # finish states
        states = np.concatenate(states)
        self.reset_sets()
        return states

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

    def step(self, agents: List[SurrogateAgent]):
        done = False
        extra_info = ''
        # allocate agents tx power
        for agent in agents:
            for pair in self.d2d_pairs:
                if agent.id == pair[0].id:
                    pair[0].set_tx_power(agent.action)
        # mue_tx_power
        mue_tx_power = self.mue.get_tx_power_db(
            self.bs, self.params.sinr_threshold, self.params.noise_power,
            self.total_losses[self.mue.id][self.bs.id],
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
        self.mue.calc_set_max_speff()
        # d2d pairs sinr
        for p in self.d2d_pairs:
            if p[0].rb == self.rb:
                _ = self.sinr_d2d(
                    p[0], p[1], list(zip(*self.d2d_pairs))[0], self.mue,
                    self.params.noise_power, self.params.user_gain
                )
                p[0].calc_set_speff()
                p[0].calc_set_max_speff()
                p[0].calc_set_avg_speff_set_link_price()
        # calculate small scale fadings and set new losses
        self.step_losses()
        # get the states
        states = self.get_state()
        rewards = self.reward_function(agents)
        # normalize rewards
        # rewards = self.normalize(rewards, self.rewards_memory,
        #                          self.min_max_scaling)
        # total reward
        self.reward = rewards
        # spectral efficiencies
        self.mue_spectral_eff = self.mue.speffs[0]
        self.d2d_spectral_eff = np.sum(
            [d[0].speffs[0] for d in self.d2d_pairs]
        )
        # end
        return states, rewards, done, extra_info

    def set_d2d_pathlosses(self):
        for tx, rx in self.d2d_pairs:
            pathloss_to_rx = self.total_losses[tx.id][rx.id]
            tx.set_pathloss_d2d(pathloss_to_rx)
            pathloss_to_bs = self.total_losses[tx.id][self.bs.id]
            tx.set_pathloss_to_bs(pathloss_to_bs)

    def set_mue_pathloss(self):
        pathloss_to_bs = self.total_losses[self.mue.id][self.bs.id]
        self.mue.set_pathloss_to_bs(pathloss_to_bs)

    def calculate_pathlosses(self):
        if self.pathlosses_are_calculated:
            raise Exception(
                'Trying to calculate pathlosses at the \
                wrong time.'
            )
        txs, rxs = zip(*self.d2d_pairs)
        txs = list(txs)
        rxs = list(rxs)
        txs.append(self.mue)
        rxs.append(self.bs)
        pathlosses = {t.id: {r.id: 0 for r in rxs} for t in txs}
        for tx in txs:
            for rx in rxs:
                if rx == self.bs:
                    channel = self.channel_to_bs
                else:
                    channel = self.channel_to_devices
                distance = euclidean(tx.position, rx.position)
                pathloss = channel.pathloss(distance)
                pathlosses[tx.id][rx.id] = pathloss
        self.pathlosses = pathlosses
        self.pathlosses_are_calculated = True

    def calculate_large_scale_fadings(self):
        if self.large_scale_fadings_are_set:
            raise Exception(
                'Trying to calculate large scale fadings at the \
                wrong time.'
            )
        txs, rxs = zip(*self.d2d_pairs)
        txs = list(txs)
        rxs = list(rxs)
        txs.append(self.mue)
        rxs.append(self.bs)
        fadings = {t.id: {r.id: 0 for r in rxs} for t in txs}
        for tx in txs:
            for rx in rxs:
                if rx == self.bs:
                    channel = self.channel_to_bs
                else:
                    channel = self.channel_to_devices
                distance = euclidean(tx.position, rx.position)
                fading = channel.large_scale(distance)
                fadings[tx.id][rx.id] = fading
        self.large_scale_fadings = fadings
        self.large_scale_fadings_are_set = True

    def calculate_small_scale_fadings(self):
        if self.small_scale_fadings_are_set:
            raise Exception(
                'Trying to calculate small scale fadings at the \
                wrong time.'
            )
        txs, rxs = zip(*self.d2d_pairs)
        txs = list(txs)
        rxs = list(rxs)
        txs.append(self.mue)
        rxs.append(self.bs)
        fadings = {t.id: {r.id: 0 for r in rxs} for t in txs}
        for tx in txs:
            for rx in rxs:
                if rx == self.bs:
                    channel = self.channel_to_bs
                else:
                    channel = self.channel_to_devices
                distance = euclidean(tx.position, rx.position)
                fading = channel.small_scale(distance)
                fadings[tx.id][rx.id] = fading
        self.small_scale_fadings = fadings
        self.small_scale_fadings_are_set = True

    def calculate_total_losses(self):
        if self.total_losses_are_set:
            raise Exception(
                'Trying to calculate total losses at the \
                wrong time.'
            )
        txs, rxs = zip(*self.d2d_pairs)
        txs = list(txs)
        rxs = list(rxs)
        txs.append(self.mue)
        rxs.append(self.bs)
        losses = {t.id: {r.id: 0 for r in rxs} for t in txs}
        for tx in txs:
            for rx in rxs:
                losses[tx.id][rx.id] = self.pathlosses[tx.id][rx.id] + \
                    self.small_scale_fadings[tx.id][rx.id] + \
                    self.large_scale_fadings[tx.id][rx.id]
        self.total_losses = losses
        self.total_losses_are_set = True

    def calculate_all_losses(self):
        self.calculate_pathlosses()
        self.calculate_large_scale_fadings()
        self.calculate_small_scale_fadings()
        self.calculate_total_losses()

    def set_all_losses(self):
        self.set_d2d_pathlosses()
        self.set_mue_pathloss()
        self.set_mue_pathlosses_to_interfering()
        self.set_d2d_pathlosses_to_interfering()

    def calc_set_all_losses(self):
        self.calculate_all_losses()
        self.set_all_losses()

    def step_losses(self):
        self.calculate_small_scale_fadings()
        self.calculate_total_losses()
        self.set_all_losses()

    def set_d2d_pathlosses_to_interfering(self):
        for tx, rx in self.d2d_pairs:
            d2d_interferers = [
                p for p in self.d2d_pairs
                if p[0].id != tx.id and p[0].rb == tx.rb
            ]
            pathlosses = {}
            for itx, irx in d2d_interferers:
                pathloss = self.total_losses[itx.id][rx.id]
                pathlosses[itx.id] = pathloss
            pathloss_to_mue = self.total_losses[self.mue.id][rx.id]
            pathlosses[self.mue.id] = pathloss_to_mue
            tx.set_pathlosses_to_interfering(pathlosses)

    def set_mue_pathlosses_to_interfering(self):
        d2d_interferers = [
            p for p in self.d2d_pairs
            if p[0].rb == self.mue.rb
        ]
        pathlosses = {}
        for itx, irx in d2d_interferers:
            pathloss = self.pathlosses[itx.id][self.bs.id]
            pathlosses[itx.id] = pathloss
            # itx.set_pathloss_to_bs(pathloss)
        self.mue.set_pathlosses_to_interfering(pathlosses)

    def set_n_d2d(self, n_d2d):
        self.n_d2d = n_d2d

    def sinr_mue(
        self, mue: mobile_user, d2d_devices: List[d2d_user],
        noise_power: float,
        bs_gain: float, user_gain: float
    ):
        mue_contrib = mue.tx_power + user_gain + bs_gain \
            - self.total_losses[mue.id][self.bs.id]
        mue.set_power_at_receiver(mue_contrib)
        d2d_interferers = [
            d for d in d2d_devices
            if d.type == d2d_node_type.TX and d.rb == mue.rb
        ]
        d2d_interferences = []
        interferences = []
        for d in d2d_interferers:
            interference = db_to_power(
                d.tx_power +
                user_gain +
                bs_gain -
                self.total_losses[d.id][self.bs.id]
            )
            d2d_interferences.append((d.id, interference))
            interferences.append(interference)
            d.set_caused_mue_interference(power_to_db(interference))
        mue.set_interferences(d2d_interferences)
        total_interference = np.sum(interferences)
        sinr = mue_contrib - power_to_db(
            db_to_power(noise_power) + total_interference
        )
        snr = mue_contrib - noise_power
        mue.set_sinr(sinr)
        mue.set_snr(snr)
        return sinr

    def sinr_d2d(self, d2d_tx: d2d_user, d2d_rx: d2d_user,
                 d2d_devices: List[d2d_user], mue: mobile_user,
                 noise_power: float, user_gain: float):
        # signal power at the D2D receiver
        d2d_loss = self.total_losses[d2d_tx.id][d2d_rx.id]
        d2d_tx_contrib = d2d_tx.tx_power - d2d_loss + 2 * user_gain
        # set the power at the receiver
        d2d_tx.set_power_at_receiver(d2d_tx_contrib)
        channel_loss_to_mue = self.total_losses[mue.id][d2d_rx.id]
        mue_interference = mue.tx_power - \
            channel_loss_to_mue + 2 * user_gain
        # set the interference caused by the MUE
        d2d_tx.set_received_mue_interference(mue_interference)
        # interfering D2D devices
        d2d_interferers = [
            d for d in d2d_devices if (
                d.id != d2d_tx.id
                and d.type == d2d_node_type.TX and d.rb == d2d_tx.rb
            )
        ]
        # interferences caused by the D2D devices
        d2d_interferences = []
        for d in d2d_interferers:
            interference = \
                db_to_power(
                    d.tx_power +
                    2 * user_gain -
                    self.total_losses[d.id][d2d_rx.id]
                )
            d2d_interferences.append((d.id,  interference))
        # set the interferences caused by the D2D devices
        if len(d2d_interferences) > 0:
            d2d_tx.set_interferences(d2d_interferences)
        else:
            raise Exception(
                'It is not possible to not have interferers.'
            )
        # total interference
        _, interferences = list(zip(*d2d_interferences))
        total_interference = np.sum(interferences)
        # d2d_tx sinr
        sinr = d2d_tx_contrib - \
            power_to_db(
                db_to_power(noise_power) +
                db_to_power(mue_interference) +
                total_interference
            )
        snr = d2d_tx_contrib - noise_power
        # set the d2d_tx sinr
        d2d_tx.set_sinr(sinr)
        d2d_tx.set_snr(snr)
        # assert not np.isnan(sinr)
        return sinr

    def reset_sets(self):
        self.small_scale_fadings_are_set = False
        self.total_losses_are_set = False
        if self.mue is not None:
            self.mue.reset_set_flags()
        for d in self.d2d_pairs:
            d[0].reset_set_flags()

    def reset_pathloss_set(self):
        self.pathlosses_are_set = False
        self.pathlosses_are_calculated = False

    def reset_all_sets(self):
        self.reset_sets()
        self.pathlosses_are_set = False
        self.pathlosses_are_calculated = False

    def reset_devices_powers(self):
        for tx, _ in self.d2d_pairs:
            tx.set_tx_power(-60)

    def reset_before_build_set(self):
        self.reset_pathloss_set()
        self.reset_devices_powers()
        self.reset_sets()

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

    def rewards_old(self, agents: List[ExternalDQNAgent]):
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

    def gabriel_reward(self, agents: SurrogateAgent) -> float:
        g1 = .2
        g2 = .01
        g3 = 6
        g4 = .005
        r1 = g1 * np.mean([p[0].sinr for p in self.d2d_pairs])
        r2 = g2 * np.sum([p[0].tx_power for p in self.d2d_pairs])
        sinr_m = self.mue.sinr
        sinr_min = self.params.sinr_threshold
        if sinr_m >= sinr_min:
            r3 = g3*(2-2**(-sinr_m+sinr_min+1))
        else:
            r3 = -2*g3+g3**(sinr_m/5)
        r4 = g4 * np.sum([a.nn_output for a in agents])
        reward = r1 + r2 + r3 + r4
        return reward

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
        self, agent: SurrogateAgent
    ) -> float:
        d2d_tx = agent.d2d_tx
        d2d_speff = d2d_tx.speffs[0]
        reward = -self.reward_penalty
        if self.mue.sinr >= self.params.sinr_threshold:
            reward = 1/self.params.c_param * d2d_speff
        return reward

    def calculate_lucas_reward(
        self, agent: ExternalDQNAgent
    ) -> float:
        C = self.params.c_param
        d2d_tx = agent.d2d_tx
        d2d_speff = d2d_tx.speffs[0]
        reward = -self.reward_penalty
        if self.mue.sinr >= self.params.sinr_threshold:
            aux = self.mue.sinr - self.params.sinr_threshold
            aux = aux if aux > 0 else 0
            reward = 1/C * d2d_speff - 1/(C**2) * (aux)
        return reward

    def calculate_individual_reward(
        self, agent: ExternalDQNAgent
    ) -> float:
        d2d_tx = agent.d2d_tx
        d2d_speff = d2d_tx.speffs[0]
        reward = -self.reward_penalty * d2d_tx.interference_contrib_pct
        if self.mue.sinr >= self.params.sinr_threshold:
            reward = 1/self.params.c_param * d2d_speff
        return reward

    def calculate_sinr_reward(
        self, **kwargs
    ):
        # m_s = db_to_power(self.mue.sinr)
        # s0 = db_to_power(self.params.sinr_threshold
        # if self.mue.sinr < self.params.sinr_threshold:
        # reward = db_to_power(m_s) - db_to_power(s0)
        #   reward =
        #   return reward
        sinrs = []
        for tx, _ in self.d2d_pairs:
            sinrs.append(tx.sinr)
        sinrs = np.sum(db_to_power(np.array(sinrs)))
        if self.mue.sinr < self.params.sinr_threshold:
            return -sinrs
        else:
            return sinrs

#     def calculate_continuous_reward(
        # self, *kwargs
    # ):
        # if self.mue.sinr <= self.params.sinr_threshold:
            # reward = db_to_power(self.mue.sinr) - \
                # db_to_power(self.params.sinr_threshold)
            # # reward = 1e-9 if reward == 0 else reward
            # # reward = power_to_db(reward)
            # # reward *= reward ** 3
        # else:
            # sinrs = np.array([p[0].sinr for p in self.d2d_pairs])
            # reward = np.sum(db_to_power(sinrs))
            # reward = 1e-9 if reward == 0 else reward
            # reward = 0.2 * power_to_db(reward)
        # # if np.isnan(reward):
        # #     print('bug')
#         return reward

    def calculate_continuous_reward(self, *kargs, **kwargs):
        gamma1 = 2.0
        # gamma2 = 1.0
        # coeff2 = gamma2 * 10
        if self.mue.sinr <= self.params.sinr_threshold:
            reward = gamma1 * (self.mue.sinr - self.params.sinr_threshold)
        else:
            sinrs = np.array([p[0].sinr for p in self.d2d_pairs])
            reward = np.sum(db_to_power(sinrs))
            if reward > 10:
                reward = power_to_db(reward)
        # if np.isnan(reward):
        #     print('bug')
        return reward

    def calculate_jain_reward(self, *kargs, **kwargs):
        gamma1 = 2.0
        gamma2 = 1
        if self.mue.sinr <= self.params.sinr_threshold:
            reward = gamma1 * (self.mue.sinr - self.params.sinr_threshold)
        else:
            sinrs = np.array([p[0].sinr for p in self.d2d_pairs])
            reward = np.sum(db_to_power(sinrs))
            if reward > 10:
                reward = power_to_db(reward)
            jain = jain_index(sinrs)
            reward *= jain ** gamma2
        # if np.isnan(reward):
        #     print('bug')
        return reward

    def calculate_speffs_reward(
        self, **kwargs
    ):
        sinrs = []
        for tx, _ in self.d2d_pairs:
            sinrs.append(tx.sinr)
        speffs = np.sum(
            np.log2(1 + db_to_power(np.array(sinrs)))
        )
        if self.mue.sinr < self.params.sinr_threshold:
            return -speffs
        else:
            return .5 * speffs

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

    def make_dummy_d2d_pair(
        self, device_height=1.5
    ) -> Tuple[d2d_user, d2d_user]:
        device = d2d_user(
            len(self.d2d_pairs),
            d2d_node_type.TX,
            self.params.p_max,
        )
        device.set_distance_to_bs(self.bs.radius)
        device.set_distance_to_mue(self.bs.radius)
        device.set_position((3 * self.bs.radius, 0, device_height))
        device.set_rb(1)
        memory_size = self.memory
        device.past_sinrs = 30 * np.ones(memory_size)
        device.past_d2d_losses = np.ones(memory_size)
        device.set_tx_power(-1000)
        # device.set_pathloss_to_bs(1000)
        device.set_distance_d2d(self.params.d2d_pair_distance)
        device.speffs = 30 * np.ones(memory_size)
        device.avg_speffs = 30 * np.ones(memory_size)
        device.link_prices = 1/30 * np.ones(memory_size)
        device.is_dummy = True
        device_rx = d2d_user(len(self.d2d_pairs), d2d_node_type.RX,
                             self.params.p_max)
        device_rx.set_position(
            (3 * self.bs.radius + self.params.d2d_pair_distance, 0, device_height)
        )
        device_rx.set_rb(1)
        device_rx.is_dummy = True
        return (device, device_rx)

    def get_pairs_positions(self):
        positions = [
            (tx.position, rx.position) for tx, rx in self.d2d_pairs
            if not (tx.is_dummy or rx.is_dummy)
        ]
        return positions

    def get_mue_position(self):
        return self.mue.position

    def state_size(self):
        # 5*n + 3
        size = 0
        n = self.params.n_d2d
        if 'positions' in self.states_options:
            # d2d devices positions and mue position
            size += 4*n + 2
        if 'sinrs' in self.states_options:
            # d2d devices sinrs and mue sinr
            size += n + 1
        if 'channels' in self.states_options:
            # channels from d2d tx to rx
            # channels from d2d tx to bs
            # channel from mue to bs
            size += 2*n + 1
        return size

    def norm_position(self, pos: float):
        r = self.params.bs_radius
        norm = (pos + r)/(2*r)
        return norm

    def whitening(self, items: np.ndarray, reference: SimpleMemory):
        mu = np.mean(reference.memory)
        var = np.var(reference.memory)
        result = (items - mu)/var
        return result

    def min_max_scaling(self, items: np.ndarray, reference: SimpleMemory):
        a_min = np.min(reference.memory)
        a_max = np.max(reference.memory)
        result = (items - a_min)/(a_max - a_min)
        return result

    def normalize(self, items: np.ndarray, reference: SimpleMemory,
                  normalization: MethodType):
        # items: np.ndarray = db_to_power(items)
        reference.push(items)
        norm_items = normalization(items, reference)
        return norm_items
