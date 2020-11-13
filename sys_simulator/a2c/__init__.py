import torch
import torch.nn as nn
from torch.functional import F
from torch.distributions import Normal, Categorical
from torch.nn.modules.container import ModuleList
from torch.nn.modules.linear import Linear
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_gae_returns(device, rewards: torch.Tensor,
                        values: torch.Tensor, gamma=0.99, lbda=0.95,):
    gae = torch.zeros(len(rewards)).to(device)
    R = torch.zeros(len(rewards)).to(device)
    values = values.detach()
    # rewards = rewards.detach()
    advantages = torch.zeros(rewards.shape, requires_grad=True).to(device)
    # returns = torch.zeros(rewards.shape).to(device).detach()
    returns = torch.zeros(rewards.shape, requires_grad=True).to(device)
    for step in reversed(range(rewards.shape[1])):
        # GAE
        delta = \
            rewards[:, step] + gamma * values[:, step + 1] - values[:, step]
        advantages[:, step] = gae = delta + gamma * lbda * gae
        # returns
        R = rewards[:, step] + gamma * R
        returns[:, step] = R
    # normalization
    # with torch.no_grad():
    #     for i in range(advantages.shape[0]):
    #         advantages[i] = (advantages[i] - torch.mean(advantages[i])) / \
    #                         (torch.std(advantages[i]) + 1e-9)
    #         returns[i] = (returns[i] - torch.mean(returns[i])) / \
    #                      (torch.std(returns[i]) + 1e-9)
    return advantages, returns


def choose_action(mu, std):
    dist = Normal(mu, std)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs,
                 hidden_size, mean=0.0, std=0.1):
        super(ActorCritic, self).__init__()
        self.mean = mean
        self.std = std
        self.device =\
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self.device)

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        ).to(self.device)

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=self.mean, std=self.std)
                nn.init.constant_(m.bias, 0.0)

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x).to(self.device)
        mu = self.actor(x.view(1, -1))
        std = self.log_std.exp().expand_as(mu).to(self.device)
        dist = Normal(mu, std)
        return dist, value


class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs,
                 hidden_size, n_hidden_layers=1):
        super(NeuralNetwork, self).__init__()
        self.device =\
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_layers = ModuleList()
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(Linear(hidden_size, hidden_size))
        self.a1 = Linear(num_inputs, hidden_size)
        self.a2 = Linear(hidden_size, int(hidden_size/2))
        self.a3 = Linear(int(hidden_size/2), int(hidden_size/4))
        self.a4 = Linear(int(hidden_size/4), num_outputs)

    def forward(self, x):
        x = self.a1(x.float())
        x = torch.relu(x)
        for a in self.hidden_layers:
            x = a(x)
            x = torch.relu(x)
        x = self.a2(x)
        x = torch.relu(x)
        x = self.a3(x)
        x = torch.relu(x)
        x = self.a4(x)
        return x


class TwoHeadedMonster(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_size: int
    ):
        super(TwoHeadedMonster, self).__init__()
        self.mu1 = Linear(num_inputs, hidden_size)
        self.mu2 = Linear(hidden_size, num_outputs)
        self.std1 = Linear(num_inputs, hidden_size)
        self.std2 = Linear(hidden_size, num_outputs)

    def forward(self, x):
        mu = self.mu1(x)
        mu = torch.relu(mu)
        mu = self.mu2(mu)
        std = self.std1(x)
        std = torch.relu(std)
        std = self.std2(std)
        return mu, std


class ActorCriticDiscrete(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        hidden_size,
        n_hidden_layers=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super(ActorCriticDiscrete, self).__init__()
        self.device = device
        self.actor = NeuralNetwork(
            num_inputs, num_outputs,
            hidden_size, n_hidden_layers).to(self.device)
        self.critic = NeuralNetwork(
            num_inputs, 1,
            hidden_size, n_hidden_layers).to(self.device)

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        probs = torch.softmax(probs.view(1, -1), dim=1)
        dist = Categorical(probs)
        return dist, value, probs


class ActorCriticContinous(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        n_actions: int,
        hidden_size: int,
        n_hidden_layers: int,
        min_output: float,
        max_output: float,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super(ActorCriticContinous, self).__init__()
        self.device = device
        self.min_output = min_output
        self.max_output = max_output
        self.common = NeuralNetwork(
            num_inputs, hidden_size//4,
            hidden_size, n_hidden_layers).to(self.device)
        self.actor = TwoHeadedMonster(
            hidden_size//4, n_actions, hidden_size//2
        ).to(self.device)
        self.critic = NeuralNetwork(
            num_inputs, 1,
            hidden_size, n_hidden_layers).to(self.device)

    def forward(self, x):
        value = self.critic(x)
        common = self.common(x)
        mu, std = self.actor(common)
        dist = Normal(mu, std)
        action = dist.sample().item()
        # action = np.clip(action, self.min_output, self.max_output)
        # action = np.clip(action, self.min_output, 1e6)
        return dist, value, action


class ActorCriticDiscreteHybrid(nn.Module):
    def __init__(self, num_inputs, num_outputs,
                 hidden_size, mean=0.0, std=0.1):
        super(ActorCriticDiscreteHybrid, self).__init__()
        self.mean = mean
        self.std = std
        self.device =\
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.common = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self.device)

        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        ).to(self.device)

    def forward(self, x):
        common = self.common(x)
        value = self.critic(common).to(self.device)
        probs = self.actor(common.view(1, -1))
        dist = Categorical(probs)
        return dist, value


class ActorLSTMDiscrete(nn.Module):
    def __init__(self, num_inputs, num_outputs,
                 hidden_size, num_rnn_layers=1):
        super(ActorLSTMDiscrete, self).__init__()
        # actor layers
        self.l1 = nn.Linear(num_inputs, hidden_size).to(device)
        self.lstm = nn.LSTM(hidden_size, hidden_size,
                            num_rnn_layers).to(device)
        self.l2 = nn.Linear(hidden_size, num_outputs).to(device)
        self.num_rnn_layers = num_rnn_layers

    def forward(self, input, hidden_h, hidden_c):
        x = self.l1(input)
        x = F.relu(x)
        x = x.view(1, 1, -1)
        x, (hidden_h, hidden_c) = self.lstm(x, (hidden_h, hidden_c))
        x = self.l2(x)
        probs = F.softmax(x, dim=1)
        # for debugging
        # if torch.sum(probs) > 1:
        #     print('problems')
        dist = Categorical(probs.view(1, -1))
        return dist, hidden_h, hidden_c

    def initHidden(self):
        hidden_h = torch.zeros(self.num_rnn_layers, 1, self.hidden_size,
                               device=self.device)
        hidden_c = torch.zeros(self.num_rnn_layers, 1,
                               self.hidden_size, device=self.device)
        return hidden_h.clone(), hidden_c.clone()


class CriticLSTM(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_rnn_layers=1):
        super(CriticLSTM, self).__init__()
        # actor layers
        self.l1 = nn.Linear(num_inputs, hidden_size).to(device)
        self.lstm = nn.LSTM(hidden_size, 1).to(device)
        self.num_rnn_layers = num_rnn_layers

    def forward(self, input, hidden_h, hidden_c):
        x = self.l1(input)
        x = F.relu(x)
        value, hidden_h, hidden_c = self.lstm(x)
        return value, hidden_h, hidden_c

    def init_hidden(self):
        hidden_h = torch.zeros(self.num_rnn_layers, 1, self.hidden_size,
                               device=self.device)
        hidden_c = torch.zeros(self.num_rnn_layers, 1,
                               self.hidden_size, device=self.device)
        return hidden_h, hidden_c


class CriticMLP(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(CriticMLP, self).__init__()
        self.l1 = nn.Linear(num_inputs, hidden_size).to(device)
        self.l2 = nn.Linear(hidden_size, hidden_size).to(device)
        self.l3 = nn.Linear(hidden_size, 1).to(device)

    def forward(self, input):
        x = self.l1(input)
        x = torch.tanh(x)
        x = self.l2(x)
        x = torch.tanh(x)
        value = self.l3(x)
        return value


class A2CLSTMDiscrete(nn.Module):
    def __init__(self, num_inputs, num_outputs,
                 hidden_size, num_rnn_layers=1):
        super(A2CLSTMDiscrete, self).__init__()
        # actor layers
        self.l1 = nn.Linear(
            num_inputs + 2*hidden_size,
            hidden_size
        ).to(device)
        self.lstm = nn.LSTM(hidden_size, hidden_size,
                            num_rnn_layers).to(device)
        self.l2 = nn.Linear(hidden_size, num_outputs).to(device)
        self.num_rnn_layers = num_rnn_layers
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(device)

    def forward(self, input, hidden_h, hidden_c):
        x = self.l1(torch.cat((input, hidden_h[0], hidden_c[0]), dim=1))
        x = F.relu(x)
        x = x.view(1, 1, -1)
        x, (hidden_h, hidden_c) = self.lstm(x, (hidden_h, hidden_c))
        x = self.l2(x)
        probs = F.softmax(x, dim=1)
        dist = Categorical(probs.view(1, -1))
        value = self.critic(input).to(device)
        return dist, value, hidden_h, hidden_c

    def initHidden(self):
        hidden_h = torch.zeros(self.num_rnn_layers, 1, self.hidden_size,
                               device=device)
        hidden_c = torch.zeros(self.num_rnn_layers, 1,
                               self.hidden_size, device=device)
        return hidden_h.clone(), hidden_c.clone()
