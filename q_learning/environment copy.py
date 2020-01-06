# multi rb - in progress

import sys
import os
lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

from devices.devices import base_station, mobile_user, d2d_user, d2d_node_type
from general import general as gen
from sinr.sinr import sinr_d2d, sinr_mue
from q_learning.agent import Agent
from typing import List
from parameters.parameters import EnvironmentParameters
from sinr.sinr import sinr_d2d, sinr_mue
from q_learning.rewards import centralized_reward

import numpy as np


class RLEnvironment:
    def __init__(self, params: EnvironmentParameters):
        self.params = params
        # TODO: há como tornar as ações contínuas? quais e quantos níveis de potência devem existir?
        # self.actions = [i*params.p_max/10 for i in range(11)]
        self.states = [0,1]
        self.total_reward = 0
        self.reward = 0
        self.done = False        
    

    def build_scenario(self, agents: List[Agent]):
        # declaring the bs, mues and d2d pairs
        self.bs = base_station((0,0), radius = self.params.bs_radius)
        self.mues = [mobile_user(x) for x in range(self.params.n_mues)]
        self.d2d_pairs = [ (d2d_user(x, d2d_node_type.TX), d2d_user(x, d2d_node_type.RX)) for x in range(self.params.n_d2d) ]
        self.rbs = [i for i in range(self.params.n_mues)]

        # distributing nodes in the bs radius
        gen.distribute_nodes(self.mues, self.bs)
        for p in self.d2d_pairs:
            gen.distribute_pair_fixed_distance( p, self.bs, self.params.d2d_pair_distance)

        # plot nodes positions
        # plot_positions(bs, mues, d2d_txs, d2d_rxs)

        # rb allocation
        # TODO: definir um metodo de alocacao de RBs. Nesta primeira simulacao, estou alocando todos para o mesmo RB. 
        # alocarei aleatoriamente nas proximas simulações
        for i in range(len(self.mues)):
            self.mues[i].set_rb(i)

        for p in self.d2d_pairs:
            p[0].set_rb(self.rbs[0])
            p[1].set_rb(self.rbs[0])

        # TODO: como determinar a potencia de transmissao do mue? vou setar pmax por enquanto
        for m in self.mues:
            m.set_tx_power(self.params.p_max)

        for i in range(self.params.n_d2d):
            agents[i].set_d2d_tx_id(self.d2d_pairs[i][0].id)

        print('SCENARIO BUILT')


    def get_state(self):
        flags = np.ones(self.params.n_mues)
        for m in self.mues:
            sinr = sinr_mue(m, list(zip(*self.d2d_pairs))[0], self.bs, self.params.noise_power, self.params.bs_gain, self.params.user_gain)
            m.set_sinr(sinr)
            if sinr < self.params.sinr_threshold:
                index = self.mues.index(m, 0)
                flags[index] = 0            
        return flags
        
    def step(self, agents: List[Agent]):
        for agent in agents:
            for pair in self.d2d_pairs:
                for device in pair:
                    if agent.id == device.id:
                        device.tx_power = agent.action

        sinr_mues = list()
        sinr_d2ds = list()
        rewards = list()

        for r in self.rbs:                        
            mue = mobile_user(-1)
            for m in self.mues:
                if m.rb == r:
                    mue = m
                    sinr_m = sinr_mue(m, list(zip(*self.d2d_pairs))[0], self.bs, self.params.noise_power, self.params.bs_gain, self.params.user_gain)
                    break

            sinr_d2d_rb = list()
            for d in list(zip(*self.d2d_pairs))[0]:                
                if d.rb == r:
                    sinr_d = sinr_d2d(d, list(zip(*self.d2d_pairs))[0], mue, self.params.noise_power, self.params.user_gain)
                    sinr_d2d_rb.append(sinr_d)

            sinr_mues.append(sinr_m)                    
            sinr_d2ds.append(sinr_d2d_rb)

        for i in range(self.rbs):
            reward = centralized_reward(sinr_mues[i], sinr_d2ds[i])
            rewards.append(reward)

        state = self.get_state()
        done = not state

        return state


    def reset(self, agents: List[Agent]):
        self.build_scenario(agents)


    # def get_actions(self):
    #     return self.actions

    





        