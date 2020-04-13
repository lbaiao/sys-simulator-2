import sys
import os
lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from devices.devices import base_station, mobile_user, d2d_user, d2d_node_type
from general import general as gen
from sinr.sinr import sinr_d2d, sinr_mue
from q_learning.agents.distanceAgent import DistanceAgent
from q_learning.environments.environment import RLEnvironment
from typing import List, Callable
from parameters.parameters import LearningParameters, EnvironmentParameters
from sinr.sinr import sinr_d2d, sinr_mue
from q_learning.rewards import centralized_reward, mod_reward
from scipy.spatial.distance import euclidean

import numpy as np


class DistanceEnvironmentMulti(RLEnvironment):
    """Same as DistanceEnvironment, but for variable multiple d2d pairs and QTensor, instead of QTable
    """
    def __init__(self, params: EnvironmentParameters, reward_function, **kwargs):
        super(DistanceEnvironmentMulti, self).__init__(params, reward_function, **kwargs)

    def build_scenario(self, agents: List[DistanceAgent]):
        # declaring the bs, mues and d2d pairs
        self.bs = base_station((0,0), radius = self.params.bs_radius)
        self.bs.set_gain(self.params.bs_gain)
        self.mue = mobile_user(0)
        self.mue.set_gain(self.params.user_gain)
        self.d2d_pairs = [ (d2d_user(x, d2d_node_type.TX), d2d_user(x, d2d_node_type.RX)) for x in range(len(agents)) ]                
        self.rb = 1
        self.distances = [1/10*i*self.bs.radius for i in range(11)]

        # distributing nodes in the bs radius and setting gains
        gen.distribute_nodes([self.mue], self.bs)
        for p in self.d2d_pairs:
            gen.distribute_pair_fixed_distance( p, self.bs, self.params.d2d_pair_distance)
            for d in p:
                d.set_distance_d2d(self.params.d2d_pair_distance)
                d.set_gain(self.params.user_gain)

        # plot nodes positions
        # plot_positions(bs, mues, d2d_txs, d2d_rxs)

        # rb allocation
        # TODO: definir um metodo de alocacao de RBs. Nesta primeira simulacao, estou alocando todos para o mesmo RB. 
        # alocarei aleatoriamente nas proximas simulações
        self.mue.set_rb(self.rb)

        for p in self.d2d_pairs:
            p[0].set_rb(self.rb)
            p[1].set_rb(self.rb)

        # TODO: como determinar a potencia de transmissao do mue? vou setar pmax por enquanto
        # self.mue.set_tx_power(self.params.p_max)
        mue_tx_power = self.mue.get_tx_power(self.bs, self.params.sinr_threshold, self.params.noise_power, self.params.mue_margin, self.params.p_max)
        self.mue.set_tx_power(mue_tx_power)

        for i in range(len(agents)):
            agents[i].set_d2d_tx_id(self.d2d_pairs[i][0].id)

        # print('SCENARIO BUILT')


    def get_state(self, agent: DistanceAgent):
        distance_index = 0

        for i in range(self.params.n_d2d):
            if agent.id == self.d2d_pairs[i][0].id:
                agent.set_distance_to_bs(self.d2d_pairs[i][0].distance_to_bs)
                break

        for i in range(len(self.distances)-1):
            if agent.distance_to_bs >= self.distances[i] and agent.distance_to_bs < self.distances[i+1]:
                distance_index = i
                break

        return distance_index


    def step(self, agents: List[DistanceAgent]):
        for agent in agents:
            for pair in self.d2d_pairs:
                if agent.id == pair[0].id:
                    pair[0].tx_power = agent.action

        mue_tx_power = self.mue.get_tx_power(self.bs, self.params.sinr_threshold, self.params.noise_power, self.params.mue_margin, self.params.p_max)
        self.mue.set_tx_power(mue_tx_power)
        self.bag.append(mue_tx_power)
        sinr_m = sinr_mue(self.mue, list(zip(*self.d2d_pairs))[0], self.bs, self.params.noise_power, self.params.bs_gain, self.params.user_gain)

        sinr_d2ds = list()
        for p in self.d2d_pairs:                
            if p[0].rb == self.rb:
                sinr_d = sinr_d2d(p[0], p[1], list(zip(*self.d2d_pairs))[0], self.mue, self.params.noise_power, self.params.user_gain)
                sinr_d2ds.append(sinr_d)

        states = [self.get_state(a) for a in agents]

        flag = True
        if sinr_m < self.params.sinr_threshold:     
            flag = False
        
        rewards, mue_se, d2d_se = self.reward_function(sinr_m, sinr_d2ds, flag, self.params.c_param)

        self.mue_spectral_eff = mue_se
        self.d2d_spectral_eff = d2d_se

        done = False
        if self.early_stop:                        
            if abs(np.sum(rewards) - self.reward) <= self.change_tolerance:
                done = True

        self.reward = np.sum(rewards)

        return states, int(flag), rewards, done
