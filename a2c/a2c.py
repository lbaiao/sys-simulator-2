import torch
import torch.nn as nn
from torch.distributions import Normal


def compute_gae_returns(device, rewards: torch.Tensor,
                        values: torch.Tensor, gamma=0.99, lbda=0.95):
    gae = torch.zeros(len(rewards)).to(device)
    R = torch.zeros(len(rewards)).to(device)
    values = values.detach()
    advantages = torch.zeros(rewards.shape).to(device)
    returns = torch.zeros(rewards.shape).to(device).detach()
    for step in reversed(range(len(rewards))):
        # GAE
        delta = \
            rewards[:, step] + gamma * values[:, step + 1] - values[:, step]
        gae = delta + gamma * lbda * gae
        advantages[:, step] = gae + values[:, step]
        # returns
        R = rewards[:, step] + gamma * R
        returns[:, step] = R
    # normalization
    for i in range(advantages.shape[0]):
        advantages[i] = (advantages[i] - torch.mean(advantages[i])) / \
                        (torch.std(advantages[i]) + 1e-9)
        returns[i] = (returns[i] - torch.mean(returns[i])) / \
                     (torch.std(returns[i]) + 1e-9)
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
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        ).to(self.device)

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
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
