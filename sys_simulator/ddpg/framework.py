import numpy as np
import torch
from sys_simulator.dqn.replay_buffer import PrioritizedReplayBuffer
from sys_simulator.ddpg import DDPG, NeuralNetwork
from torch.optim import Adam


class Framework:
    def __init__(
        self,
        replay_memory_size: int,
        replay_initial: int,
        state_size: int,
        action_size: int,
        learning_rate: float,
        batch_size: int,
        gamma: float,
        polyak: float,
        beta_its: int,
        alpha: float,
        beta: float,
        device: torch.device
    ):
        self.device = device
        # replay memory
        self.replay_memory = PrioritizedReplayBuffer(
            replay_memory_size, alpha, beta, beta_its
        )
        self.replay_initial = replay_initial
        # ddpg
        self.ddpg = DDPG(state_size, action_size)
        self.ddpg.to(device)
        self.target_ddpg = \
            DDPG(state_size, action_size, True)
        self.target_ddpg.to(device)
        # clone ddpg NNs to the target
        self.target_ddpg.actor.load_state_dict(self.ddpg.actor.state_dict())
        self.target_ddpg.critic.load_state_dict(self.ddpg.critic.state_dict())
        self.actor_optimizer = \
            Adam(self.ddpg.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = \
            Adam(self.ddpg.critic.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.gamma = gamma
        self.polyak = polyak

    def learn(self):
        if len(self.replay_memory) < self.replay_initial:
            return
        obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes = \
            self.replay_memory.sample(self.batch_size)
        obses_t = torch.tensor(
            obses_t, dtype=torch.float, device=self.device
        ).view(obses_t.shape[0], -1)
        actions = torch.tensor(
            actions, dtype=torch.float, device=self.device
        ).view(-1, 1)
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=self.device
        ).view(-1, 1)
        obses_tp1 = torch.tensor(
            obses_tp1, dtype=torch.float, device=self.device
        ).view(obses_tp1.shape[0], -1)
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        # targets
        actions_tp1 = self.target_ddpg.actor(obses_tp1)
        q_tp1 = self.target_ddpg.critic(obses_tp1, actions_tp1)
        aux = torch.tensor(1 - dones).view(-1, 1).float().to(self.device)
        targets = rewards + self.gamma * aux * q_tp1
        # train critic
        q_tp = self.ddpg.critic(obses_t, actions)
        critic_losses = torch.pow((q_tp - targets), 2)
        critic_loss = critic_losses.mean()
        critic_loss.backward()
        self.critic_optimizer.step()
        # train actor
        actions_tp = self.ddpg.actor(obses_t)
        q_tp_actor = self.ddpg.critic(obses_t, actions_tp)
        actor_loss = q_tp_actor.mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.polyak_update(self.target_ddpg.critic, self.ddpg.critic)
        self.polyak_update(self.target_ddpg.actor, self.ddpg.actor)
        # update experience replay
        prios = np.multiply(
            weights,
            critic_losses.clone().detach().cpu().numpy().reshape(-1)
        )
        prios += 1e-5
        self.replay_memory.update_priorities(batch_idxes, prios)
        self.replay_memory.update_beta()

    def polyak_update(self, target_nn: NeuralNetwork, nn: NeuralNetwork):
        with torch.no_grad():
            for p_, p in zip(
                target_nn.parameters(),
                nn.parameters()
            ):
                p_ = self.polyak * p_ + (1 - self.polyak) * p
