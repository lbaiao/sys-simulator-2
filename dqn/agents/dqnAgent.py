import os
import sys
lucas_path = os.environ['LUCAS_PATH']
sys.path.insert(1, lucas_path)

import numpy as np
from dqn.dqn import DQN
from parameters.parameters import DQNAgentParameters
from q_learning.agents.agent import Agent
from dqn.replayMemory import ReplayMemory, Transition
from dqn.dqn import DQN
import torch

class DQNAgent(Agent):
    """
    don't forget to set the agent actions with the set_actions method
    """
    def __init__(self, params: DQNAgentParameters, actions):
        super(DQNAgent, self).__init__(params, actions)
        self.batchsize = params.batchsize
        self.gamma = params.gamma
        self.replay_memory = ReplayMemory(10000)        
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.device = torch.device("cuda")        

        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.criterion = torch.nn.SmoothL1Loss

    def set_distance_to_bs(self, distance: float):
        self.distance_to_bs = distance
    
    def get_action(self, obs):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() > self.epsilon:
            # aux = torch.tensor([obs[0]], device=self.device)
            self.action = torch.tensor(self.policy_net(obs[0]), device=self.device).max(1)[1][0]
        else:
            self.action = torch.tensor(np.random.choice([i for i in range(len(self.actions))])).cpu()
            self.action = torch.tensor(self.action, device=self.device)
        return self.action

    def learn(self):
        if len(self.replay_memory) < self.batchsize:
            return
        transitions = self.replay_memory.sample(self.batchsize)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.tensor(batch.state, device=self.device).reshape(self.batchsize, batch.state[0].shape[1]).float()
        next_state_batch = torch.tensor(batch.next_state, device=self.device).reshape(self.batchsize, batch.next_state[0].shape[1]).float()
        action_batch = torch.tensor(batch.action, device=self.device).reshape(self.batchsize, 1).float()
        reward_batch = torch.tensor(batch.reward, device=self.device).reshape(self.batchsize, 1).float()

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.long())
        next_state_values = torch.zeros(self.batchsize, device=self.device)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)

        expected_state_action_values = next_state_values * self.gamma + reward_batch        

        loss = self.criterion(state_action_values.float(), expected_state_action_values.float())

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()


        