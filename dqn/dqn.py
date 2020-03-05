import os
import random
import torch
import numpy as np
from pprint import pprint
from collections import namedtuple
from parameters.parameters import AgentParameters
from typing import List

# see: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# class DQN(torch.nn.Module):
#     """    
#     """
#     def __init__(self):
#         super(DQN, self).__init__()
#         self.linear1 = torch.nn.Linear(5,8).cuda()
#         self.linear2 = torch.nn.Linear(8,16).cuda()
#         self.linear3 = torch.nn.Linear(16,16).cuda()
#         self.linear4 = torch.nn.Linear(16,16).cuda()
#         self.linear5 = torch.nn.Linear(16,16).cuda()
#         self.linear6 = torch.nn.Linear(16,16).cuda()
#         self.linear7 = torch.nn.Linear(16,21).cuda()

#     def forward(self, x):
#         h_tanh1 = self.linear1(x).relu().cuda()
#         h_tanh2 = self.linear2(h_tanh1).relu().cuda()
#         h_tanh3 = self.linear3(h_tanh2).relu().cuda()
#         h_tanh4 = self.linear4(h_tanh3).relu().cuda()
#         h_tanh5 = self.linear5(h_tanh4).relu().cuda()
#         h_tanh6 = self.linear6(h_tanh5).relu().cuda()        
#         y_pred = self.linear7(h_tanh6).softmax(1).cuda()
#         self.q_values = h_tanh6
#         return y_pred


# class DQN(torch.nn.Module):
#     """    
#     """
#     def __init__(self):
#         super(DQN, self).__init__()
#         self.linear1 = torch.nn.Linear(5,8).cuda()
#         self.linear2 = torch.nn.Linear(8,16).cuda()
#         self.linear3 = torch.nn.Linear(16,32).cuda()
#         self.linear4 = torch.nn.Linear(32,64).cuda()
#         self.linear5 = torch.nn.Linear(64,32).cuda()
#         self.linear6 = torch.nn.Linear(32,32).cuda()
#         self.linear7 = torch.nn.Linear(32,21).cuda()

#     def forward(self, x):
#         h_tanh1 = self.linear1(x).relu().cuda()
#         h_tanh2 = self.linear2(h_tanh1).relu().cuda()
#         h_tanh3 = self.linear3(h_tanh2).relu().cuda()
#         h_tanh4 = self.linear4(h_tanh3).relu().cuda()
#         h_tanh5 = self.linear5(h_tanh4).relu().cuda()
#         h_tanh6 = self.linear6(h_tanh5).relu().cuda()        
#         y_pred = self.linear7(h_tanh6).softmax(1).cuda()
#         self.q_values = h_tanh6
#         return y_pred


class DQN(torch.nn.Module):
    """    
    """
    def __init__(self):
        super(DQN, self).__init__()
        self.linear1 = torch.nn.Linear(5,8).cuda()
        self.linear2 = torch.nn.Linear(8,16).cuda()
        self.linear3 = torch.nn.Linear(16,32).cuda()
        self.linear4 = torch.nn.Linear(32,64).cuda()
        self.linear5 = torch.nn.Linear(64,64).cuda()
        self.linear6 = torch.nn.Linear(64,64).cuda()
        self.linear7 = torch.nn.Linear(64,64).cuda()
        self.linear8 = torch.nn.Linear(64,64).cuda()
        self.linear10 = torch.nn.Linear(64,32).cuda()
        self.linear11 = torch.nn.Linear(32,32).cuda()
        self.linear12 = torch.nn.Linear(32,21).cuda()

    def forward(self, x):
        h_tanh1 = self.linear1(x).relu().cuda()
        h_tanh2 = self.linear2(h_tanh1).relu().cuda()
        h_tanh3 = self.linear3(h_tanh2).relu().cuda()
        h_tanh4 = self.linear4(h_tanh3).relu().cuda()
        h_tanh5 = self.linear5(h_tanh4).relu().cuda()
        h_tanh6 = self.linear6(h_tanh5).relu().cuda()        
        h_tanh7 = self.linear7(h_tanh6).relu().cuda()        
        h_tanh8 = self.linear8(h_tanh7).relu().cuda()        
        h_tanh10 = self.linear10(h_tanh8).relu().cuda()        
        h_tanh11 = self.linear11(h_tanh10).relu().cuda()            
        y_pred = self.linear12(h_tanh11).softmax(dim=1).cuda()
        self.q_values = h_tanh11
        return y_pred
