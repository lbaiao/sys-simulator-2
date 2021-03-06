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
    ):
        super(DDPGCritic, self).__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class DDPG(Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        is_target=False
    ):
        super(DDPG, self).__init__()
        self.actor = DDPGActor(state_size, action_size)
        self.critic = DDPGCritic(state_size, action_size)
        if is_target:
            self.actor.eval()
            self.critic.eval()
