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
# import pandas as pd
import torch


class CompleteEnvironment(RLEnvironment):
    """
    Environment implemented for solving the power allocation problem, for D2D devices in underlay mode, in the uplink,
    using a Deep Q-Learning algorithm. The returned states are based on agent conditions, in each RB, and consists of: number of d2d
    users in the same RB, agent distance to BS, distance between d2d_rx and MUE in the same RB, distance between MUE and BS,
    distance between d2d agent tx and nearest other pair d2d rx, interference indicator.
    """
    def __init__(self, params: EnvironmentParameters, reward_function, **kwargs):
        self.params = params
        # TODO: há como tornar as ações contínuas? quais e quantos níveis de potência devem existir?
        # self.actions = [i*params.p_max/10 for i in range(11)]
        super(CompleteEnvironment, self).__init__(params, reward_function, **kwargs)
        self.states = [0,0,1]
        self.device = torch.device('cuda')
    

    def build_scenario(self, agents: List[DistanceAgent]):
        # declaring the bs, mues and d2d pairs
        self.bs = base_station((0,0), radius = self.params.bs_radius)
        self.bs.set_gain(self.params.bs_gain)
        self.mue = mobile_user(0)
        self.mue.set_gain(self.params.user_gain)
        self.d2d_pairs = [ (d2d_user(x, d2d_node_type.TX), d2d_user(x, d2d_node_type.RX)) for x in range(self.params.n_d2d) ]
        self.rb = 1
        self.distances = [1/10*i*self.bs.radius for i in range(11)]

        # distributing nodes in the bs radius        
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
        self.mue.set_tx_power(self.params.p_max)

        for i in range(self.params.n_d2d):
            agents[i].set_d2d_tx_id(self.d2d_pairs[i][0].id)

        # print('SCENARIO BUILT')


    def get_state(self, agent: DistanceAgent):
        flag = 1
        sinr = sinr_mue(self.mue, list(zip(*self.d2d_pairs))[0], self.bs, self.params.noise_power, self.params.bs_gain, self.params.user_gain)
        distance_index = 0
        mue_distance_index = 0

        (index, d2d_tx) = [(index, p[0]) for index, p in enumerate(self.d2d_pairs) if p[0].id == agent.id][0]                
        d2d_rx = self.d2d_pairs[index][1]

        number_of_d2d_pairs = len(self.d2d_pairs)
        d2d_tx_distance_to_bs = d2d_tx.distance_to_bs
        d2d_rx_distance_to_mue = euclidean(d2d_rx.position, self.mue.position)
        mue_distance_to_bs = self.mue.distance_to_bs

        d2d_tx_distance_to_nearest = 1e9
        for i, p in enumerate(self.d2d_pairs):
            if i != index:
                dist = euclidean(d2d_tx.position, p[1].position)
                if dist < d2d_tx_distance_to_nearest:
                    d2d_tx_distance_to_nearest = dist
             
        interference_indicator = int(sinr > self.params.sinr_threshold)
        
        state = torch.tensor([[number_of_d2d_pairs, d2d_tx_distance_to_bs, d2d_rx_distance_to_mue, mue_distance_to_bs, interference_indicator]], device=self.device)
        # state = pd.DataFrame(state, columns=['number_of_d2d_pairs', 'd2d_tx_distance_to_bs', 'd2d_rx_distance_to_mue', 'mue_distance_to_bs', 'interference_indicator'])        

        return state


    def step(self, agents: List[DistanceAgent]):
        for agent in agents:
            for pair in self.d2d_pairs:
                if agent.id == pair[0].id:
                    pair[0].tx_power = agent.action

        mue_tx_power = self.mue.get_tx_power(self.bs, self.params.sinr_threshold, self.params.noise_power, self.params.mue_margin, self.params.p_max)
        self.mue.set_tx_power(mue_tx_power)
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

        done = False
        if self.early_stop:                        
            if torch.abs(torch.sum(rewards) - self.reward) <= self.change_tolerance:
                done = True

        self.reward = torch.sum(rewards)

        self.mue_spectral_eff.append(mue_se)
        self.d2d_spectral_eff.append(d2d_se)        

        return states, rewards, done
    
    def get_state_index(self, agent_distance: int, mue_distance: int, interference: bool):
        if interference:
            state_index = 10*agent_distance + mue_distance + 100
            return state_index
        else:
            state_index = 10*agent_distance + mue_distance
            return state_index


        

