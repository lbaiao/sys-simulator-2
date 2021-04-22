from types import MethodType
from sys_simulator.dqn.dqn import DQN, NoisyDuelingDQN, SimpleNN
from sys_simulator.parameters.parameters import DQNAgentParameters
from sys_simulator.dqn.replay_buffer \
    import PrioritizedReplayBuffer, ReplayBuffer
import torch
import numpy as np


class ExternalDQNFramework:
    unpack_batch: MethodType

    def __init__(self, params: DQNAgentParameters,
                 input_size: int, output_size: int, hidden_size: int,
                 device: torch.device,
                 n_hidden_layers=5, learning_rate=1e-3,
                 replay_memory_type='standard', **kwargs):
        self.device = device
        # replay memory
        self.replay_memory_type = replay_memory_type
        self.replay_memory = ReplayBuffer(params.replay_memory_size)
        if replay_memory_type == 'standard':
            self.replay_memory = ReplayBuffer(params.replay_memory_size)
            self.unpack_batch = self.unpack_std
        elif replay_memory_type == 'prioritized':
            alpha = kwargs['alpha']
            beta = kwargs['beta']
            beta_its = kwargs['beta_its']
            self.replay_memory = PrioritizedReplayBuffer(
                params.replay_memory_size, alpha, beta, beta_its
            )
            self.unpack_batch = self.unpack_prio
        self.policy_net = SimpleNN(
            input_size, output_size, hidden_size, n_hidden_layers
        ).to(self.device)
        self.target_net = SimpleNN(
            input_size, output_size, hidden_size, n_hidden_layers
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # self.optimizer = torch.optim.RMSprop(
        #     self.policy_net.parameters(), lr=learning_rate
        # )
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.criterion = torch.nn.MSELoss()
        self.batchsize = params.batchsize
        self.gamma = params.gamma
        self.bag = list()

    def learn(self):
        if len(self.replay_memory) < self.batchsize:
            return float('nan')
        obses_t, actions, rewards, obses_tp1, dones, extra_args = \
            self.unpack_batch()
        obses_t = torch.tensor(
            obses_t, dtype=torch.float, device=self.device
        )
        actions = torch.tensor(
            actions, dtype=torch.float, device=self.device
        )
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=self.device
        )
        obses_tp1 = torch.tensor(
            obses_tp1, dtype=torch.float, device=self.device
        )
        dones = torch.tensor(
            dones, dtype=torch.float, device=self.device
        )
        q_tp = \
            self.policy_net(obses_t).gather(1, actions.view(-1, 1).long())
        # metrics, q values average
        # self.bag.append(torch.mean(q_tp).item())
        q_tp1 = \
            self.target_net(obses_tp1).max(1)[0].detach().unsqueeze(1)
        targets = \
            q_tp1 * (1.0 - dones) * self.gamma + rewards
        losses = torch.pow((q_tp - targets.detach()), 2)
        if self.replay_memory_type == 'prioritized':
            losses *= torch.FloatTensor(extra_args['weights'])\
                .view(-1, 1).to(self.device)
        loss = losses.mean()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.replay_memory_type == 'prioritized':
            weights = extra_args['weights']
            batch_idxes = extra_args['batch_idxes']
            prios = np.multiply(
                weights,
                losses.clone().detach().cpu().numpy().reshape(-1)
            )
            prios += 1e-5
            self.replay_memory.update_priorities(batch_idxes, prios)
            self.replay_memory.update_beta()
        return loss.item()

    def unpack_std(self):
        obses_t, actions, rewards, obses_tp1, dones = \
            self.replay_memory.sample(self.batchsize)
        return obses_t, actions, rewards, obses_tp1, dones, {}

    def unpack_prio(self):
        obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes = \
            self.replay_memory.sample(self.batchsize)
        kwargs = {'weights': weights, 'batch_idxes': batch_idxes}
        return obses_t, actions, rewards, obses_tp1, dones, kwargs


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
            self.unpack_batch()
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
