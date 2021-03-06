import torch
import math
from sys_simulator.a2c import \
    A2C, ActorCriticContinous, ActorCriticDiscrete, compute_gae_returns


class Framework:
    a2c: A2C
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    entropy: torch.Tensor
    n_environments: int
    steps_per_episode: int
    beta: float
    device: torch.device

    def __init__(
        self,
        n_environments: int,
        steps_per_episode: int,
        beta: float,
        gamma: float,
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    ):
        self.device = device
        self.beta = beta
        self.gamma = gamma
        self.n_environments = n_environments
        self.steps_per_episode = steps_per_episode
        self.criterion = torch.nn.MSELoss()
        self.log_probs = \
            torch.zeros((self.n_environments, self.steps_per_episode))\
            .float().to(self.device)
        self.values = \
            torch.zeros((self.n_environments, self.steps_per_episode+1))\
            .float().to(self.device)
        self.rewards = \
            torch.zeros((self.n_environments, self.steps_per_episode))\
            .float().to(self.device)
        self.entropy = \
            torch.zeros((self.n_environments, self.steps_per_episode))\
            .float().to(self.device)

    def reset_values(self, n_environments):
        self.log_probs = \
            torch.zeros((n_environments, self.steps_per_episode))\
            .float().to(self.device)
        self.values = \
            torch.zeros((n_environments, self.steps_per_episode+1))\
            .float().to(self.device)
        self.rewards = \
            torch.zeros((n_environments, self.steps_per_episode))\
            .float().to(self.device)
        self.entropy = \
            torch.zeros((n_environments, self.steps_per_episode))\
            .float().to(self.device)

    def learn(self):
        # gae and returns
        advantages, returns = \
            compute_gae_returns(
                self.device, self.rewards, self.values, self.gamma
            )
        # update critic
        critic_loss = self.criterion(
            self.values[:, :-1].reshape(1, -1),
            returns.view(1, -1)
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # for param in self.a2c.critic.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()
        # update actor
        actor_loss = torch.mul(advantages, self.log_probs)
        actor_loss -= self.beta * self.entropy
        actor_loss = torch.sum(actor_loss, axis=1)
        actor_loss = -torch.mean(actor_loss)
        # actor_loss = -torch.mean(actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        for param in self.a2c.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()
        # reset values
        self.reset_values(self.n_environments)


class DiscreteFramework(Framework):
    a2c: ActorCriticDiscrete
    criterion: torch.nn.MSELoss
    actor_optimizer: torch.optim.SGD
    critic_optimizer: torch.optim.SGD

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        num_environment: int,
        steps_per_episode: int,
        learning_rate: float,
        beta: float,
        gamma: float,
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    ):
        super(DiscreteFramework, self).__init__(
            num_environment, steps_per_episode, beta, gamma, device
        )
        self.a2c = ActorCriticDiscrete(
            input_size,
            output_size,
            hidden_size,
            n_hidden_layers,
            device
        )
        self.actor_optimizer = \
            torch.optim.SGD(self.a2c.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = \
            torch.optim.SGD(self.a2c.critic.parameters(), lr=learning_rate)


class ContinuousFramework(Framework):
    a2c: ActorCriticContinous
    criterion: torch.nn.MSELoss
    actor_optimizer: torch.optim.SGD
    critic_optimizer: torch.optim.SGD

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        min_output: float,
        max_output: float,
        steps_per_episode: int,
        learning_rate: float,
        beta: float,
        gamma: float,
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    ):
        super(ContinuousFramework, self).__init__(
            1, steps_per_episode, beta, gamma, device
        )
        self.mu = torch.zeros((1, steps_per_episode)).to(device)
        self.actions = torch.zeros((1, steps_per_episode)).to(device)
        self.vars = torch.zeros((1, steps_per_episode)).to(device)
        self.a2c = ActorCriticContinous(
            input_size,
            output_size,
            hidden_size,
            n_hidden_layers,
            min_output,
            max_output,
            device
        )
        self.actor_optimizer = \
            torch.optim.SGD(self.a2c.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = \
            torch.optim.SGD(self.a2c.critic.parameters(), lr=learning_rate)
        self.steps_per_episode = steps_per_episode
        self.device = device

    def learn(self):
        self.log_probs = calc_logprob(self.mu, self.vars, self.actions)
        self.entropy = -(torch.log(2*math.pi*self.vars) + 1)/2
        super(ContinuousFramework, self).learn()
        self.mu = \
            torch.zeros((1, self.steps_per_episode)).to(self.device)
        self.actions = \
            torch.zeros((1, self.steps_per_episode)).to(self.device)
        self.vars = \
            torch.zeros((1, self.steps_per_episode)).to(self.device)


def calc_logprob(mu_v, var_v, actions_v):
    p1 = -((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-3))
    p2 = -torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2
