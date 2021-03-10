import math
import random
import torch
from torch.nn \
    import Linear, ModuleList, Parameter, ReLU, Softmax, Module
from torch.functional import F
from collections import namedtuple
from torch.nn.modules.container import Sequential

from torch.nn.modules.module import Module

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
    """Script 32
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers=1
    ):
        super(DQN, self).__init__()
        self.hidden_layers = ModuleList()
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(Linear(hidden_size, hidden_size))
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, int(hidden_size/2))
        self.fc3 = Linear(int(hidden_size/2), int(hidden_size/4))
        self.fc4 = Linear(int(hidden_size/4), output_size)

    def forward(self, obs):
        x = self.fc1(obs)
        x = torch.tanh(x)
        # x = torch.dropout(x, .2, True)
        for i in self.hidden_layers:
            x = i(x)
            x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        x = self.fc4(x)
        # output = F.softmax(x, dim=1)
        output = x
        return output


class SimpleNN(Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers=1
    ):
        super(SimpleNN, self).__init__()
        self.hidden_layers = ModuleList()
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(Linear(hidden_size, hidden_size))
        self.fc1 = Linear(input_size, hidden_size)
        self.fc_out = Linear(hidden_size, output_size)

    def forward(self, obs):
        x = self.fc1(obs)
        x = torch.relu(x)
        # x = torch.dropout(x, .2, True)
        for i in self.hidden_layers:
            x = i(x)
            x = torch.relu(x)
        x = self.fc_out(x)
        # output = F.softmax(x, dim=1)
        output = x
        return output


class NoisyLinear(Linear):
    def __init__(self, in_features, out_features,
                 sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * \
                self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + \
            self.weight
        return F.linear(input, v, bias)


class NoisyDQN(DQN):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers=1
    ):
        super(NoisyDQN, self).__init__(
            input_size, output_size,
            hidden_size, n_hidden_layers
        )
        self.fc3 = NoisyLinear(
            int(hidden_size/2),
            int(hidden_size/4)
        )
        self.fc4 = NoisyLinear(
            int(hidden_size/4),
            output_size
        )


class NoisyDuelingDQN(Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers=1
    ):
        super(NoisyDuelingDQN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        # layers
        self.hidden_layers = ModuleList()
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(Linear(hidden_size, hidden_size))
        self.fc1 = Linear(input_size, hidden_size)
        # advantages
        self.fc_adv = Sequential(
            NoisyLinear(hidden_size, hidden_size),
            ReLU(),
            NoisyLinear(hidden_size, output_size)
        )
        # value
        self.fc_val = Sequential(
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, 1)
        )

    def forward(self, obs):
        x = self.fc1(obs)
        x = torch.relu(x)
        for i in self.hidden_layers:
            x = i(x)
            x = torch.tanh(x)
        adv = self.fc_adv(x)
        val = self.fc_val(x)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q
