import torch
from torch.nn import Module, Linear
from torch import nn
from torch.nn.modules.container import ModuleList
from torch import tanh


class NeuralNetwork(Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers=1
    ):
        super(NeuralNetwork, self).__init__()
        self.hidden_layers = ModuleList()
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(Linear(hidden_size, hidden_size))
        self.a1 = Linear(input_size, hidden_size)
        self.a2 = Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.a1(x.float())
        x = tanh(x)
        for a in self.hidden_layers:
            x = a(x)
            x = tanh(x)
        x = self.a2(x)
        return x


class DDPGCritic(nn.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        h1: int,
        h2: int,
        init_w=3e-4
    ):
        super(DDPGCritic, self).__init__()
        self.linear1 = Linear(obs_size + act_size, h1)
        self.linear2 = Linear(h1, h2)
        self.linear3 = Linear(h2, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x, a):
        x = torch.relu(self.linear1(torch.cat([x, a], dim=1)))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DDPGActor(nn.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        h1: int,
        h2: int,
        init_w=3e-3
    ):
        super(DDPGActor, self).__init__()
        self.linear1 = Linear(obs_size, h1)
        self.linear2 = Linear(h1, h2)
        self.linear3 = Linear(h2, act_size)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class DDPG:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        h1: int,
        h2: int,
        device: torch.device
    ):
        self.actor = DDPGActor(state_size, action_size, h1, h2).to(device)
        self.critic = DDPGCritic(state_size, action_size, h1, h2).to(device)
