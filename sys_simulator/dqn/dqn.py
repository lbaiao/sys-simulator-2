import math
import random
import torch
from torch.nn import Linear, ModuleList, Parameter, ReLU, Softmax
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
#     """ Script 15
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


# class DQN(torch.nn.Module):
#     """ Script 16, script19
#     """
#     def __init__(self):
#         super(DQN, self).__init__()
#         self.fc1 = torch.nn.Linear(7,6).cuda()
#         self.fc2 = torch.nn.Linear(6,6).cuda()
#         self.fc3 = torch.nn.Linear(6,6).cuda()
#         self.fc4 = torch.nn.Linear(6,6).cuda()
#         self.fc5 = torch.nn.Linear(6,5).cuda()


#     def forward(self, state):
#         x = self.fc1(state).tanh().cuda()
#         x = self.fc2(x).tanh().cuda()
#         x = self.fc3(x).tanh().cuda()
#         # x = torch.nn.Dropout(0.2)(x)
#         x = self.fc4(x).tanh().cuda()
#         y = self.fc5(x).cuda()

#         return y


# class DQN(torch.nn.Module):
#     """ Script 17, 18
#     """
#     def __init__(self):
#         super(DQN, self).__init__()
#         self.linear1 = torch.nn.Linear(5,8)
#         self.linear2 = torch.nn.Linear(8,16)
#         self.linear3 = torch.nn.Linear(16,32)
#         self.linear4 = torch.nn.Linear(32,64)
#         self.linear5 = torch.nn.Linear(64,32)
#         self.linear6 = torch.nn.Linear(32,32)
#         self.linear7 = torch.nn.Linear(32,5)

#     def forward(self, x):
#         h_tanh1 = self.linear1(x).relu()
#         h_tanh2 = self.linear2(h_tanh1).relu()
#         h_tanh3 = self.linear3(h_tanh2).relu()
#         h_tanh4 = self.linear4(h_tanh3).relu()
#         h_tanh5 = self.linear5(h_tanh4).relu()
#         h_tanh6 = self.linear6(h_tanh5).relu()
#         y_pred = self.linear7(h_tanh6).softmax(1)
#         self.q_values = h_tanh6
#         return y_pred


# class DQN(torch.nn.Module):
#     """ Script2_3, Script2_4
#     """
#     def __init__(self):
#         super(DQN, self).__init__()
#         self.fc1 = torch.nn.Linear(2,5).cuda()
#         self.fc2 = torch.nn.Linear(5,5).cuda()
#         self.fc3 = torch.nn.Linear(5,5).cuda()
#         self.fc4 = torch.nn.Linear(5,5).cuda()
#         self.fc5 = torch.nn.Linear(5,5).cuda()


#     def forward(self, state):
#         x = self.fc1(state).tanh().cuda()
#         x = self.fc2(x).tanh().cuda()
#         x = self.fc3(x).tanh().cuda()
#         x = self.fc4(x).tanh().cuda()
#         y = self.fc5(x).cuda()

#         return y


# class DQN(torch.nn.Module):
#     """ script23, 26, 27, 28
#     """
#     def __init__(self):
#         super(DQN, self).__init__()
#         self.fc1 = torch.nn.Linear(6, 5).cuda()
#         self.fc2 = torch.nn.Linear(5, 5).cuda()
#         self.fc3 = torch.nn.Linear(5, 5).cuda()
#         self.fc4 = torch.nn.Linear(5, 5).cuda()
#         self.fc5 = torch.nn.Linear(5, 5).cuda()

#     def forward(self, state):
#         x = self.fc1(state).tanh().cuda()
#         x = self.fc2(x).tanh().cuda()
#         x = self.fc3(x).tanh().cuda()
#         # x = torch.nn.Dropout(0.2)(x)
#         x = self.fc4(x).tanh().cuda()
#         y = self.fc5(x).cuda()

#         return y


# class DQN(torch.nn.Module):
#     """ Script 24
#     """
#     def __init__(self):
#         super(DQN, self).__init__()
#         self.fc1 = torch.nn.Linear(8,7).cuda()
#         self.fc2 = torch.nn.Linear(7,7).cuda()
#         self.fc3 = torch.nn.Linear(7,7).cuda()
#         self.fc4 = torch.nn.Linear(7,7).cuda()
#         self.fc5 = torch.nn.Linear(7,5).cuda()


#     def forward(self, state):
#         x = self.fc1(state).tanh().cuda()
#         x = self.fc2(x).tanh().cuda()
#         x = self.fc3(x).tanh().cuda()
#         # x = torch.nn.Dropout(0.2)(x)
#         x = self.fc4(x).tanh().cuda()
#         y = self.fc5(x).cuda()

#         return y


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
