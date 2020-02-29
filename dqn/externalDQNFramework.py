from dqn.dqn import DQN
from dqn.replayMemory import ReplayMemory, Transition
from parameters.parameters import DQNAgentParameters
import torch

class ExternalDQNFramework:
    def __init__(self, params: DQNAgentParameters):
        self.replay_memory = ReplayMemory(20000)        
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.device = torch.device("cuda")        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.criterion = torch.nn.SmoothL1Loss()
        self.batchsize = params.batchsize
        self.gamma = params.gamma
        self.bag = list()

    def learn(self):
        if len(self.replay_memory) < self.batchsize:
            return
        transitions = self.replay_memory.sample(self.batchsize)
        batch = Transition(*zip(*transitions))
        
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

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()