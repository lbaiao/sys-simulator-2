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
from dqn.externalDQNFramework import ExternalDQNFramework
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
        self.criterion = torch.nn.SmoothL1Loss()

    def set_distance_to_bs(self, distance: float):
        self.distance_to_bs = distance
    
    def get_action(self, obs):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() > self.epsilon:
            # aux = torch.tensor([obs[0]], device=self.device)
            self.action_index = torch.tensor(self.policy_net(obs), device=self.device).max(1)[1][0]
            self.action = self.actions[self.action_index]
        else:
            self.action_index = torch.tensor(np.random.choice([i for i in range(len(self.actions))])).cpu()
            self.action_index = torch.tensor(self.action_index, device=self.device)
            self.action = self.actions[self.action_index]
        return self.action
    

    def set_policy(self, policy: DQN):
        self.policy_net = policy


    def act(self, obs: torch.Tensor):
        return self.policy_net(obs)


    def set_action(self, action_index: torch.Tensor, action: torch.Tensor):
        self.action_index = action_index.long()
        self.action = action


    def learn(self):
        if len(self.replay_memory) < self.batchsize:
            return
        transitions = self.replay_memory.sample(self.batchsize)
        batch = Transition(*zip(*transitions))

        self.optimizer.zero_grad()
        
        state_batch = torch.zeros([self.batchsize, batch.state[0].shape[1]], device=self.device)
        torch.cat(batch.state, out=state_batch)
        next_state_batch = torch.zeros([self.batchsize, batch.state[0].shape[1]], device=self.device)        
        torch.cat(batch.next_state, out=next_state_batch)
        action_batch = torch.tensor(batch.action, device=self.device).reshape(self.batchsize, 1).float()
        reward_batch = torch.tensor(batch.reward, device=self.device).reshape(self.batchsize, 1).float()

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.long())
        self.bag.append(torch.mean(self.policy_net.q_values)) # metrics, q values average
        # self.bag.append(torch.mean(self.policy_net.q_values[0,:])) # metrics, first q value average
        next_state_values = torch.zeros(self.batchsize, device=self.device)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)

        expected_state_action_values = next_state_values * self.gamma + reward_batch        

        loss = self.criterion(state_action_values.float(), expected_state_action_values.float())

        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()


class ExternalDQNAgent(Agent):
    """
    don't forget to set the agent actions with the set_actions method
    Same as DQNAgent, but the agent does not have its own DQN.
    """
    def __init__(self, params: DQNAgentParameters, actions):
        super(ExternalDQNAgent, self).__init__(params, actions)
        self.device = torch.device("cuda")        

    def set_distance_to_bs(self, distance: float):
        self.distance_to_bs = distance
    
    def get_action(self, policy: ExternalDQNFramework, obs):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() > self.epsilon:
            # aux = torch.tensor([obs[0]], device=self.device)
            self.action_index = policy.policy_net(obs).max(1)[1][0]
            self.action = self.actions[self.action_index]
        else:
            self.action_index = torch.tensor(np.random.choice([i for i in range(len(self.actions))])).cpu()
            self.action_index = torch.tensor(self.action_index, device=self.device)
            self.action = self.actions[self.action_index]
        return self.action
    

    def act(self, framework: ExternalDQNFramework, obs: torch.Tensor):
        return framework.policy_net(obs)


    def set_action(self, action_index: torch.Tensor, action: torch.Tensor):
        self.action_index = action_index.long()
        self.action = action    
    

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
