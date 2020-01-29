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


class MUEDistanceEnvironment(RLEnvironment):
    """
    Environment implemented for the Q Learning Based Power Control algorithms found in 
    Nie, S., Fan, Z., Zhao, M., Gu, X. and Zhang, L., 2016, September. Q-learning based power control algorithm for D2D communication. 
    In 2016 IEEE 27th Annual International Symposium on Personal, Indoor, and Mobile Radio Communications 
    (PIMRC) (pp. 1-6). IEEE.
    THIS ENVIRONMENT ONLY WORKS IF THE DISTANCES FROM AGENT TO BS AND MUE TO BS ARE DIVIDED INTO 10 ZONES, WHICH GIVES A 10*10*2 POSSIBLE STATES.
    """
    def __init__(self, params: EnvironmentParameters, reward_function, **kwargs):
        self.params = params
        # TODO: há como tornar as ações contínuas? quais e quantos níveis de potência devem existir?
        # self.actions = [i*params.p_max/10 for i in range(11)]
        super(MUEDistanceEnvironment, self).__init__(params, reward_function, **kwargs)
        self.states = [0,0,1]
    

    def build_scenario(self, agents: List[DistanceAgent]):
        # declaring the bs, mues and d2d pairs
        self.bs = base_station((0,0), radius = self.params.bs_radius)
        self.mue = mobile_user(0)
        self.d2d_pairs = [ (d2d_user(x, d2d_node_type.TX), d2d_user(x, d2d_node_type.RX)) for x in range(self.params.n_d2d) ]
        self.rb = 1
        self.distances = [1/10*i*self.bs.radius for i in range(11)]

        # distributing nodes in the bs radius        
        gen.distribute_nodes([self.mue], self.bs)
        for p in self.d2d_pairs:
            gen.distribute_pair_fixed_distance( p, self.bs, self.params.d2d_pair_distance)
            for d in p:
                d.set_distance_d2d(self.params.d2d_pair_distance)

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

        for i in range(self.params.n_d2d):
            if agent.id == self.d2d_pairs[i][0].id:
                agent.set_distance_to_bs(self.d2d_pairs[i][0].distance_to_bs)
                break

        for i in range(len(self.distances)-1):
            if agent.distance_to_bs >= self.distances[i] and agent.distance_to_bs < self.distances[i+1]:
                distance_index = i
                break

        for i in range(len(self.distances)-1):
            if self.mue.distance_to_bs >= self.distances[i] and agent.distance_to_bs < self.distances[i+1]:
                mue_distance_index = i
                break

        if sinr < self.params.sinr_threshold:            
            flag = 0
        
        return self.get_state_index(distance_index, mue_distance_index, flag)


    def step(self, agents: List[DistanceAgent]):
        for agent in agents:
            for device in self.d2d_pairs[0]:
                if agent.id == device.id:
                    device.tx_power = agent.action

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
            if abs(np.sum(rewards) - self.reward) <= self.change_tolerance:
                done = True

        self.reward = np.sum(rewards)

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


        

