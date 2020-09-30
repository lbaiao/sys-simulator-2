from sys_simulator.dqn.dqn import DQN
from sys_simulator.dqn.replayMemory import ReplayMemory, Transition
from sys_simulator.parameters.parameters import DQNAgentParameters
import torch


class ExternalDQNFramework:
    def __init__(self, params: DQNAgentParameters,
                 input_size: int, output_size: int, hidden_size: int,
                 n_hidden_layers=5):
        self.replay_memory = ReplayMemory(params.replay_memory_size)
        self.device = torch.device("cuda")
        self.policy_net = DQN(
            input_size, output_size, hidden_size, n_hidden_layers
        ).to(self.device)
        self.target_net = DQN(
            input_size, output_size, hidden_size, n_hidden_layers
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.SGD(
            self.policy_net.parameters(), lr=0.01
        )
        self.criterion = torch.nn.MSELoss()
        self.batchsize = params.batchsize
        self.gamma = params.gamma
        self.bag = list()

    def learn(self):
        if len(self.replay_memory) < self.batchsize:
            return
        transitions = self.replay_memory.sample(self.batchsize)
        batch = Transition(*zip(*transitions))

        self.optimizer.zero_grad()

        state_batch = torch.zeros(
            [self.batchsize, batch.state[0].shape[1]],
            device=self.device
        )
        torch.cat(batch.state, out=state_batch)
        next_state_batch = torch.zeros(
            [self.batchsize, batch.state[0].shape[1]],
            device=self.device
        )
        torch.cat(batch.next_state, out=next_state_batch)
        action_batch = torch.tensor(
            batch.action,
            device=self.device
        ).reshape(self.batchsize, 1).float()
        reward_batch = torch.tensor(
            batch.reward,
            device=self.device
        ).reshape(self.batchsize, 1).float()
        state_action_values = \
            self.policy_net(state_batch).gather(1, action_batch.long())
        # metrics, q values average
        self.bag.append(torch.mean(state_action_values))
        next_state_values = torch.zeros(self.batchsize, device=self.device)
        next_state_values = \
            self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
        expected_state_action_values = \
            next_state_values * self.gamma + reward_batch
        loss = self.criterion(
            state_action_values.float(),
            expected_state_action_values.float()
        )
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
