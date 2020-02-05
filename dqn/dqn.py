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


class DQN(torch.nn.Module):
    """
    
    """
    def __init__(self):
        super(DQN, self).__init__()
        self.linear1 = torch.nn.Linear(5,5)
        self.linear2 = torch.nn.Linear(5,5)
        self.linear3 = torch.nn.Linear(5,5)
        self.linear4 = torch.nn.Linear(5,5)
        self.linear5 = torch.nn.Linear(5,5)
        self.linear6 = torch.nn.Linear(5,5)
        self.linear7 = torch.nn.Linear(5,11)

    def forward(self, x):
        h_tanh1 = self.linear1(x).relu()
        h_tanh2 = self.linear2(h_tanh1).relu()
        h_tanh3 = self.linear3(h_tanh2).relu()
        h_tanh4 = self.linear4(h_tanh3).relu()
        h_tanh5 = self.linear5(h_tanh4).relu()
        h_tanh6 = self.linear6(h_tanh5).relu()
        y_pred = self.linear7(h_tanh6).softmax(1)       
        return y_pred


class Agent:
    def __init__(self, params: AgentParameters, actions: List[int]):
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.epsilon = 1
        self.actions = actions 
        self.replayMemory = ReplayMemory(replayMemorySize)
        self.batchSize = int(batchSize)

        self.policyNet = DQN()
        self.targetNet = DQN()
        self.memory = ReplayMemory(10000)
        self.targetNet.load_state_dict(self.policyNet.state_dict())        
        self.targetNet.eval()

        self.optimizer = torch.optim.Adam(self.policyNet.parameters())
        self.criterion = torch.nn.SmoothL1Loss()

    def getAction(self,obs):
        if self.epsilon > self.epsilonMin:
            self.epsilon -= self.epsilonDecay
        if np.random.random() > self.epsilon:
            try:
                aux = torch.tensor([obs, torch.tensor(0)], device=self.device).reshape(2,1)                
                return torch.tensor(self.policyNet(aux.float(), device=self.device).max(1)[1][0])
            except Exception as e:
                print(e)
        else:
            return torch.tensor(np.random.choice([ a for a in range(len(self.actionVector))]), device=self.device)

    def learn(self,obs,action,reward,netObs):
        pass # train the dqn

    def optimizeModel(self):
        if len(self.replayMemory) < self.batchSize:
            return
        transitions = self.replayMemory.sample(self.batchSize)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple([int(s) is not None for s in batch.next_state])).reshape(self.batchSize,1).float()
        # non_final_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_states = torch.tensor([s for s in batch.next_state if s is not None]).reshape(self.batchSize, 1).float()

        state_batch = torch.tensor(batch.state).reshape(self.batchSize,1).float()
        action_batch = torch.tensor(batch.action).reshape(self.batchSize,1).float()
        reward_batch = torch.tensor(batch.reward).reshape(self.batchSize,1).float()

        state_action_values = self.policyNet(state_batch).gather(1, action_batch.long()) 

        next_state_values = torch.zeros(self.batchSize)
        next_state_values[non_final_mask.long()] = self.targetNet(non_final_states).max(1)[0].detach().unsqueeze(1)

        expected_state_action_values = (next_state_values.unsqueeze(1) * self.gamma) + reward_batch
        expected_state_action_values = expected_state_action_values.long()

        loss = self.criterion(state_action_values.float(), expected_state_action_values.float())

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policyNet.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()



