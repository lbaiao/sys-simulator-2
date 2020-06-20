import torch
import torch.nn as nn
from torch.distributions import Normal


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


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
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu).to(self.device)
        dist = Normal(mu, std)
        return dist, value
