from types import MethodType
import numpy as np
import torch
from torch.nn.modules.module import Module
from sys_simulator.dqn.replay_buffer \
    import PrioritizedReplayBuffer, ReplayBuffer
from sys_simulator.ddpg import DDPGActor, DDPGCritic
from torch.optim import Adam
from torch.nn import MSELoss
# import random


class Framework:
    replay_memory_size: int
    replay_initial: int
    state_size: int
    action_size: int
    actor_learning_rate: float
    critic_learning_rate: float
    batch_size: int
    gamma: float
    soft_tau: float
    replay_memory: str
    device: torch.device
    unpack_batch: MethodType

    def __init__(
        self,
        replay_memory_type: str,
        replay_memory_size: int,
        replay_initial: int,
        state_size: int,
        action_size: int,
        hidden_size_1: int,
        hidden_size_2: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        batch_size: int,
        gamma: float,
        soft_tau: float,
        device: torch.device,
        **kwargs
    ):
        self.device = device
        # replay memory
        self.replay_memory_type = replay_memory_type
        if replay_memory_type == 'standard':
            self.replay_memory = ReplayBuffer(replay_memory_size)
            self.unpack_batch = self.unpack_std
        elif replay_memory_type == 'prioritized':
            alpha = kwargs['alpha']
            beta = kwargs['beta']
            beta_its = kwargs['beta_its']
            self.replay_memory = PrioritizedReplayBuffer(
                replay_memory_size, alpha, beta, beta_its
            )
            self.unpack_batch = self.unpack_prio
        else:
            raise Exception('Invalid experience replay.')
        self.replay_initial = replay_initial
        # ddpg
        self.actor = DDPGActor(
            state_size, action_size,
            hidden_size_1, hidden_size_2).to(device)
        self.critic = DDPGCritic(
            state_size, action_size,
            hidden_size_1, hidden_size_2).to(device)
        self.target_actor = DDPGActor(
            state_size, action_size,
            hidden_size_1, hidden_size_2).to(device)
        self.target_critic = DDPGCritic(
            state_size, action_size,
            hidden_size_1, hidden_size_2).to(device)
        self.target_actor.eval()
        self.target_critic.eval()
        # clone ddpg NNs to the target
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = \
            Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = \
            Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.critic_criterion = MSELoss()
        self.batch_size = batch_size
        self.gamma = gamma
        self.soft_tau = soft_tau

    def learn(self):
        if len(self.replay_memory) < self.replay_initial \
                or len(self.replay_memory) < self.batch_size:
            return float('-inf'), float('inf')
        obses_t, actions, rewards, obses_tp1, dones, extra_args = \
            self.unpack_batch()
        obses_t = torch.tensor(
            obses_t, dtype=torch.float, device=self.device
        ).view(obses_t.shape[0], -1)
        actions = torch.tensor(
            actions, dtype=torch.float, device=self.device
        ).view(-1, actions.shape[2])
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=self.device
        ).view(-1, 1)
        obses_tp1 = torch.tensor(
            obses_tp1, dtype=torch.float, device=self.device
        ).view(obses_tp1.shape[0], -1)
        dones = torch.tensor(
            dones, dtype=torch.float, device=self.device
        ).view(obses_tp1.shape[0], -1)
        # train actor
        actions_tp = self.actor(obses_t)
        q_tp_actor = self.critic(obses_t, actions_tp)
        actor_loss = -q_tp_actor.mean()
        # targets
        actions_tp1 = self.target_actor(obses_tp1)
        q_tp1 = self.target_critic(obses_tp1, actions_tp1.detach())
        targets = rewards + (1.0 - dones) * self.gamma * q_tp1
        # train critic
        q_tp = self.critic(obses_t, actions)
        critic_losses = torch.pow((q_tp - targets.detach()), 2)
        # critic_losses = self.critic_criterion(q_tp, targets.detach())
        critic_loss = critic_losses.mean()
        # updates
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.soft_tau_update(self.target_critic, self.critic)
        self.soft_tau_update(self.target_actor, self.actor)
        # update experience replay
        if self.replay_memory_type == 'prioritized':
            weights = extra_args['weights']
            batch_idxes = extra_args['batch_idxes']
            prios = np.multiply(
                weights,
                critic_losses.clone().detach().cpu().numpy().reshape(-1)
            )
            prios += 1e-5
            self.replay_memory.update_priorities(batch_idxes, prios)
            self.replay_memory.update_beta()
        return actor_loss.item(), critic_loss.item()

    def soft_tau_update(self, target_nn: Module, nn: Module):
        # with torch.no_grad():
        for p_, p in zip(
            target_nn.parameters(),
            nn.parameters()
        ):
            with torch.no_grad():
                p_.data.copy_(
                    p_.data * (1 - self.soft_tau) + p.data * self.soft_tau
                )

    def copy_nn(self, target_nn: Module, nn: Module):
        for p_, p in zip(
            target_nn.parameters(),
            nn.parameters()
        ):
            with torch.no_grad():
                p_.data.copy_(p.data)

    def unpack_std(self):
        obses_t, actions, rewards, obses_tp1, dones = \
            self.replay_memory.sample(self.batch_size)
        return obses_t, actions, rewards, obses_tp1, dones, {}

    def unpack_prio(self):
        obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes = \
            self.replay_memory.sample(self.batch_size)
        kwargs = {'weights': weights, 'batch_idxes': batch_idxes}
        return obses_t, actions, rewards, obses_tp1, dones, kwargs

