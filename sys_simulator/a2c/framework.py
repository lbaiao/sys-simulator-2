import torch
from sys_simulator.a2c import \
    ActorCriticContinous, ActorCriticDiscrete, compute_gae_returns


class Framework:
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    entropy: torch.Tensor
    steps_per_episode: int
    beta: int
    device: torch.device

    def __init__(
        self,
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
        self.steps_per_episode = steps_per_episode
        self.criterion = torch.nn.MSELoss()
        self.log_probs = torch.zeros((1, steps_per_episode)).to(device)
        self.values = torch.zeros((1, steps_per_episode+1)).to(device)
        self.rewards = torch.zeros((1, steps_per_episode)).to(device)
        self.entropy = torch.zeros((1, steps_per_episode)).to(device)

    def learn(self):
        # gae and returns
        advantages, returns = \
            compute_gae_returns(
                self.device, self.rewards, self.values, self.gamma
            )
        # update critic
        critic_loss = self.criterion(self.values[0][:-1], returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # update actor
        actor_loss = torch.mul(advantages, self.log_probs)
        actor_loss -= self.beta * self.entropy
        actor_loss = -torch.sum(actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # reset values
        self.log_probs = \
            torch.zeros((1, self.steps_per_episode)).float().to(self.device)
        self.values = \
            torch.zeros((1, self.steps_per_episode+1)).float().to(self.device)
        self.rewards = \
            torch.zeros((1, self.steps_per_episode)).float().to(self.device)
        self.entropy = \
            torch.zeros((1, self.steps_per_episode)).float().to(self.device)


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
        steps_per_episode: int,
        learning_rate: float,
        beta: float,
        gamma: float,
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    ):
        super(DiscreteFramework, self).__init__(
            steps_per_episode, beta, gamma, device
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
            steps_per_episode, beta, gamma, device
        )
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
