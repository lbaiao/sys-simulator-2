import numpy as np
from torch.optim import SGD, AdamW
import torch
import math
from sys_simulator.a2c import \
    A2C, ActorCritic, ActorCriticContinous, ActorCriticDiscrete, compute_gae, \
    compute_gae_returns


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
            torch.optim.Adam(self.a2c.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = \
            torch.optim.Adam(self.a2c.critic.parameters(), lr=learning_rate)


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
            SGD(self.a2c.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = \
            SGD(self.a2c.critic.parameters(), lr=learning_rate)
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


class PPOFramework():
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        steps_per_episode: int,
        n_envs: int,
        actor_lr: float,
        critic_lr: float,
        beta=.001,
        gamma=.99,
        lbda=.95,
        clip_param=.2,
        optimizers='adam',
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    ):
        self.device = device
        self.beta = beta
        self.gamma = gamma
        self.lbda = lbda
        self.clip_param = clip_param
        self.steps_per_episode = steps_per_episode
        self.n_envs = n_envs
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.values = []
        self.log_probs = []
        self.entropy = 0
        self.next_state = 0
        self.next_value = 0
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.a2c = ActorCritic(
            input_size,
            output_size,
            hidden_size,
            n_hidden_layers,
        ).to(device)
        if optimizers == 'adam':
            self.a_optim = AdamW(self.a2c.actor.parameters(), lr=actor_lr)
            self.c_optim = AdamW(self.a2c.critic.parameters(), lr=critic_lr)
        elif optimizers == 'sgd':
            self.a_optim = SGD(self.a2c.actor.parameters(), lr=actor_lr)
            self.c_optim = SGD(self.a2c.critic.parameters(), lr=critic_lr)
        else:
            raise Exception('Invalid optimizer.')

    def reset_values(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.values = []
        self.log_probs = []
        self.next_state = 0
        self.next_value = 0
        self.entropy = 0

    def ppo_iter(self, mini_batch_size, states, actions, log_probs,
                 returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], \
                log_probs[rand_ids, :], returns[rand_ids, :], \
                advantage[rand_ids, :]

    def learn(self, ppo_epochs, mini_batch_size):
        advantages, returns = compute_gae_returns(
            self.next_value, self.rewards,
            self.masks, self.values, self.gamma, self.lbda)
        returns = torch.cat(returns).detach()
        log_probs = torch.cat(self.log_probs).detach()
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        advantages = torch.cat(advantages).detach()
        losses = []
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in \
                    self.ppo_iter(mini_batch_size, states, actions,
                                  log_probs, returns, advantages):
                dist, value = self.a2c(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)
                # importance sampling
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * advantage
                # losses
                actor_loss = \
                    -torch.min(surr1, surr2).mean() - self.beta * entropy
                critic_loss = (return_ - value).pow(2).mean()
                # update
                self.c_optim.zero_grad()
                critic_loss.backward()
                self.c_optim.step()
                self.a_optim.zero_grad()
                actor_loss.backward()
                self.a_optim.step()
                self.reset_values()
                losses.append((actor_loss.item(), critic_loss.item()))
        self.reset_values()
        return losses

    # def learn(self, ppo_epochs, mini_batch_size):
    #     returns = compute_gae(self.next_value, self.rewards,
    #                           self.masks, self.values, self.gamma, self.lbda)
    #     returns = torch.cat(returns).detach()
    #     log_probs = torch.cat(self.log_probs).detach()
    #     values = torch.cat(self.values).detach()
    #     states = torch.cat(self.states)
    #     actions = torch.cat(self.actions)
    #     advantages = returns - values
    #     advantages = returns
    #     losses = []
    #     for _ in range(ppo_epochs):
    #         for state, action, old_log_probs, return_, advantage in \
    #                 self.ppo_iter(mini_batch_size, states, actions,
    #                               log_probs, returns, advantages):
    #             dist, value = self.a2c(state)
    #             entropy = dist.entropy().mean()
    #             new_log_probs = dist.log_prob(action)
    #             # importance sampling
    #             ratio = (new_log_probs - old_log_probs).exp()
    #             surr1 = ratio * advantage
    #             surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
    #                                 1.0 + self.clip_param) * advantage
    #             # losses
    #             actor_loss = -torch.min(surr1, surr2).mean()
    #             critic_loss = (return_ - value).pow(2).mean()
    #             loss = 0.5 * critic_loss + actor_loss - self.beta * entropy
    #             # update
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #             losses.append(loss.item())
    #     self.reset_values()
    #     return losses

    def push_experience(self, log_prob, value, reward, done, state, action):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(
            torch.FloatTensor(reward)
            .view(self.n_envs, -1).to(self.device))
        self.masks.append(
            torch.FloatTensor(1 - done)
            .view(self.n_envs, -1).to(self.device))
        self.states.append(
            torch.FloatTensor(state)
            .view(self.n_envs, -1).to(self.device))
        self.actions.append(
            torch.FloatTensor(action)
            .view(self.n_envs, -1).to(self.device))

    def push_next(self, next_state, next_value, entropy):
        self.next_state = \
            torch.FloatTensor(next_state)\
            .view(self.n_envs, -1).to(self.device)
        self.next_value = next_value
        self.entropy = entropy


class A2CFramework(PPOFramework):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        steps_per_episode: int,
        n_envs: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        beta=.001,
        gamma=.99,
        lbda=.95,
        optimizers='adam',
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    ):
        super(A2CFramework, self).__init__(
            input_size,
            output_size,
            hidden_size,
            n_hidden_layers,
            steps_per_episode,
            n_envs,
            actor_learning_rate,
            critic_learning_rate,
            beta,
            gamma,
            lbda,
            .2,
            optimizers,
            device
        )
        if optimizers == 'adam':
            self.a_optim = \
                AdamW(self.a2c.actor.parameters(), lr=actor_learning_rate)
            self.c_optim = \
                AdamW(self.a2c.critic.parameters(), lr=critic_learning_rate)
        elif optimizers == 'sgd':
            self.a_optim = \
                SGD(self.a2c.actor.parameters(), lr=actor_learning_rate)
            self.c_optim = \
                SGD(self.a2c.critic.parameters(), lr=critic_learning_rate)
        else:
            raise Exception('Invalid optimizer.')

    def learn(self):
        advantages, returns = compute_gae_returns(
            self.next_value, self.rewards,
            self.masks, self.values, self.gamma, self.lbda)
        advantages = torch.cat(advantages).view(-1, self.n_envs).detach()
        returns = torch.cat(returns).detach()
        log_probs = torch.cat(self.log_probs).view(-1, self.n_envs)
        values = torch.cat(self.values)
        entropy = self.entropy
        # losses
        actor_loss = -(log_probs * advantages).sum(dim=0)
        actor_loss = actor_loss.mean() - self.beta * entropy
        critic_loss = (values - returns).pow(2).mean()
        # update
        self.c_optim.zero_grad()
        critic_loss.backward()
        self.c_optim.step()
        self.a_optim.zero_grad()
        actor_loss.backward()
        self.a_optim.step()
        self.reset_values()
        return actor_loss.item(), critic_loss.item()


class A2CDiscreteFramework(A2CFramework):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        steps_per_episode: int,
        n_envs: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        beta=.001,
        gamma=.99,
        lbda=.95,
        optimizers='adam',
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    ):
        super(A2CDiscreteFramework, self).__init__(
            input_size,
            output_size,
            hidden_size,
            n_hidden_layers,
            steps_per_episode,
            n_envs,
            actor_learning_rate,
            critic_learning_rate,
            beta,
            gamma,
            lbda,
            optimizers,
            device
        )
        self.a2c = ActorCriticDiscrete(
            input_size, output_size, hidden_size,
            n_hidden_layers
        ).to(device)
        if optimizers == 'adam':
            self.a_optim = \
                AdamW(self.a2c.actor.parameters(), lr=actor_learning_rate)
            self.c_optim = \
                AdamW(self.a2c.critic.parameters(), lr=critic_learning_rate)
        elif optimizers == 'sgd':
            self.a_optim = \
                SGD(self.a2c.actor.parameters(), lr=actor_learning_rate)
            self.c_optim = \
                SGD(self.a2c.critic.parameters(), lr=critic_learning_rate)
        else:
            raise Exception('Invalid optimizer.')

    def push_experience(self, log_prob, value, reward, done, state, action):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(
            torch.FloatTensor(reward)
            .view(self.n_envs, -1).to(self.device))
        self.masks.append(
            torch.FloatTensor(1 - done)
            .view(self.n_envs, -1).to(self.device))
        self.states.append(
            torch.FloatTensor(state)
            .view(self.n_envs, -1).to(self.device))
        self.actions.append(
            torch.LongTensor(action.cpu()))

    def learn(self):
        advantages, returns = \
            compute_gae_returns(
                self.next_value, self.rewards,
                self.masks, self.values, self.gamma, self.lbda)
        advantages = torch.cat(advantages).view(-1, self.n_envs).detach()
        returns = torch.cat(returns).detach()
        log_probs = torch.cat(self.log_probs).view(-1, self.n_envs)
        values = torch.cat(self.values)
        entropy = self.entropy
        # losses
        actor_loss = -(log_probs * advantages).sum(dim=0)
        actor_loss = actor_loss.mean() - self.beta * entropy
        critic_loss = (values - returns).pow(2).mean()
        # update
        self.c_optim.zero_grad()
        critic_loss.backward()
        self.c_optim.step()
        self.a_optim.zero_grad()
        actor_loss.backward()
        self.a_optim.step()
        self.reset_values()
        return actor_loss.item(), critic_loss.item()
