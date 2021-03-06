from sys_simulator.dqn.dqn import DQN, NoisyDuelingDQN
from sys_simulator.dqn.replayMemory import ReplayMemory, Transition
from sys_simulator.parameters.parameters import DQNAgentParameters
from sys_simulator.dqn.replay_buffer import PrioritizedReplayBuffer
import torch
import torch.nn.functional as F
import numpy as np


class ExternalDQNFramework:
    def __init__(self, params: DQNAgentParameters,
                 input_size: int, output_size: int, hidden_size: int,
                 n_hidden_layers=5, learning_rate=.5):
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
        # self.optimizer = torch.optim.RMSprop(
        #     self.policy_net.parameters(), lr=learning_rate
        # )
        self.optimizer = torch.optim.SGD(
            self.policy_net.parameters(), lr=learning_rate
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
        self.bag.append(torch.mean(state_action_values).item())
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


class RainbowFramework:
    def __init__(self, params: DQNAgentParameters,
                 input_size: int, output_size: int, hidden_size: int,
                 beta_its: int,
                 n_hidden_layers=5, learning_rate=.5,
                 alpha=.6, beta=.4):
        self.replay_memory = PrioritizedReplayBuffer(
            params.replay_memory_size, alpha,
            beta, beta_its
        )
        self.device = torch.device("cuda")
        self.policy_net = NoisyDuelingDQN(
            input_size, output_size, hidden_size,
            n_hidden_layers
        ).to(self.device)
        self.target_net = NoisyDuelingDQN(
            input_size, output_size, hidden_size,
            n_hidden_layers
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # self.optimizer = torch.optim.RMSprop(
        #     self.policy_net.parameters(), lr=learning_rate
        # )
        self.optimizer = torch.optim.SGD(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.criterion = torch.nn.MSELoss()
        self.batchsize = params.batchsize
        self.gamma = params.gamma
        self.bag = list()

    def learn(self):
        if len(self.replay_memory) < self.batchsize:
            return
        obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes = \
            self.replay_memory.sample(self.batchsize)
        obses_t = torch.tensor(
            obses_t, dtype=torch.float, device=self.device
        ).view(obses_t.shape[0], -1)
        actions = torch.tensor(
            actions, dtype=torch.long, device=self.device
        ).view(-1, 1)
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=self.device
        ).view(-1, 1)
        obses_tp1 = torch.tensor(
            obses_tp1, dtype=torch.float, device=self.device
        ).view(obses_tp1.shape[0], -1)
        self.optimizer.zero_grad()
        # Q(s,a)
        state_actions_vals = self.policy_net(obses_t).gather(1, actions)
        state_actions_vals = state_actions_vals.view(-1, 1)
        # Q(s',a')
        next_states_v = self.target_net(obses_tp1).max(1)[0].view(-1, 1)
        exp_sa_vals = next_states_v.detach() * self.gamma + rewards
        losses = torch.pow(state_actions_vals - exp_sa_vals, 2)
        # TD-errors
        prios = np.multiply(
            weights,
            losses.clone().detach().cpu().numpy().reshape(-1)
        )
        prios += 1e-5
        loss = losses.mean()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.replay_memory.update_priorities(batch_idxes, prios)
        self.replay_memory.update_beta()
