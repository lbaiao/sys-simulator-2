import sys
import os
# lucas_path = os.environ['LUCAS_PATH']
# sys.path.insert(1, lucas_path)

from sys_simulator.devices.devices import base_station, mobile_user, d2d_user, d2d_node_type
from sys_simulator.general import general as gen
from sys_simulator.sinr.sinr import sinr_d2d, sinr_mue
from sys_simulator.q_learning.agents.agent import Agent
from typing import List, Callable
from sys_simulator.parameters.parameters import EnvironmentParameters
from sys_simulator.sinr.sinr import sinr_d2d, sinr_mue
from sys_simulator.q_learning.rewards import centralized_reward, mod_reward

import numpy as np

class RLEnvironment:
    """
    Environment implemented for the Q Learning Based Power Control algorithms found in 
    Nie, S., Fan, Z., Zhao, M., Gu, X. and Zhang, L., 2016, September. Q-learning based power control algorithm for D2D communication. 
    In 2016 IEEE 27th Annual International Symposium on Personal, Indoor, and Mobile Radio Communications 
    (PIMRC) (pp. 1-6). IEEE.    
    """
    def __init__(self, params: EnvironmentParameters, reward_function, **kwargs):
        """
        done_disable: 'early_stop'
        """
        self.params = params
        # TODO: há como tornar as ações contínuas? quais e quantos níveis de potência devem existir?
        # self.actions = [i*params.p_max/10 for i in range(11)]
        self.states = [0,1]
        self.total_reward = 0
        self.reward = 0
        self.old_reward = 0
        self.done = False        
        self.mue_spectral_eff = list()
        self.d2d_spectral_eff = list()
        self.reward_function = reward_function
        self.early_stop = False
        self.tolerance = 0  
        self.tolerance_count = 0        
        self.bag = list()      

        if 'early_stop' in kwargs:            
            self.early_stop = True
            self.change_tolerance = kwargs['early_stop']
            if 'tolerance' in kwargs:
                self.tolerance = kwargs['tolerance']                


    def build_scenario(self, agents: List[Agent]):
        # declaring the bs, mues and d2d pairs
        self.bs = base_station((0,0), radius = self.params.bs_radius)
        self.mue = mobile_user(0)
        self.d2d_pairs = [ (d2d_user(x, d2d_node_type.TX), d2d_user(x, d2d_node_type.RX)) for x in range(self.params.n_d2d) ]
        self.rb = 1

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


    def get_state(self):
        flag = 1
        sinr = sinr_mue(self.mue, list(zip(*self.d2d_pairs))[0], self.bs, self.params.noise_power, self.params.bs_gain, self.params.user_gain)
        if sinr < self.params.sinr_threshold:
            flag = 0
        return flag


    def step(self, agents: List[Agent]):
        for agent in agents:
            for pair in self.d2d_pairs:
                if agent.id == pair[0].id:
                    pair[0].tx_power = agent.action

        sinr_m = sinr_mue(self.mue, list(zip(*self.d2d_pairs))[0], self.bs, self.params.noise_power, self.params.bs_gain, self.params.user_gain)

        sinr_d2ds = list()
        for p in self.d2d_pairs:                
            if p[0].rb == self.rb:
                sinr_d = sinr_d2d(p[0], p[1], list(zip(*self.d2d_pairs))[0], self.mue, self.params.noise_power, self.params.user_gain)
                sinr_d2ds.append(sinr_d)

        state = self.get_state() 

        # reward, mue_se, d2d_se = centralized_reward(sinr_m, sinr_d2ds)
        # reward, mue_se, d2d_se = mod_reward(sinr_m, sinr_d2ds, state)
        reward, mue_se, d2d_se = self.reward_function(sinr_m, sinr_d2ds, state)

        self.mue_spectral_eff.append(mue_se)
        self.d2d_spectral_eff.append(d2d_se)

        done = False
        if self.early_stop:                        
            if abs(reward - self.reward) <= self.change_tolerance:                
                self.tolerance_count += 1
                if self.tolerance_count >= self.tolerance:
                    done = True
                    self.tolerance_count = 0
            
        self.old_reward = reward

        return state, reward, done


    def reset(self, agents: List[Agent]):
        self.d2d_spectral_eff = list()
        self.mue_spectral_eff = list()
        self.build_scenario(agents)


    # def get_actions(self):
    #     return self.actions

    





        