from sys_simulator.devices.devices import d2d_user
import numpy as np
from sys_simulator.dqn.dqn import DQN
from sys_simulator.parameters.parameters import DQNAgentParameters
from sys_simulator.q_learning.agents.agent import Agent
from sys_simulator.dqn.replayMemory import ReplayMemory, Transition
from sys_simulator.dqn.externalDQNFramework import ExternalDQNFramework, RainbowFramework
import torch


class DQNAgent(Agent):
    """
    don't forget to set the agent actions with the set_actions method
    """

    def __init__(self, params: DQNAgentParameters, actions):
        super(DQNAgent, self).__init__(params, actions)
        self.batchsize = params.batchsize
        self.gamma = params.gamma
        self.replay_memory = ReplayMemory(1000)
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.device = torch.device("cuda")

        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.criterion = torch.nn.NLLLoss()

    def set_distance_to_bs(self, distance: float):
        self.distance_to_bs = distance

    def get_action(self, obs):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() > self.epsilon:
            # aux = torch.tensor([obs[0]], device=self.device)
            self.action_index = torch.tensor(self.policy_net(obs),
                                             device=self.device).max(1)[1][0]
            self.action = self.actions[self.action_index]
        else:
            self.action_index = torch.tensor(
                np.random.choice([i for i in range(len(self.actions))])
            ).cpu()
            self.action_index = torch.tensor(self.action_index,
                                             device=self.device)
            self.action = self.actions[self.action_index]
        return self.action

    def set_policy(self, policy: DQN):
        self.policy_net = policy

    def act(self, obs: torch.Tensor):
        return self.policy_net(obs)

    def set_action(self, action_index: torch.Tensor, action: torch.Tensor):
        self.action_index = action_index.long()
        self.action = action

    def learn(self):
        if len(self.replay_memory) < self.batchsize:
            return
        transitions = self.replay_memory.sample(self.batchsize)
        batch = Transition(*zip(*transitions))
        self.optimizer.zero_grad()
        state_batch = torch.zeros(
            [self.batchsize, batch.state[0].shape[1]],
            device=self.device
        )
        torch.cat(batch.state, out=state_batch)
        next_state_batch = torch.zeros(
            [self.batchsize, batch.state[0].shape[1]],
            device=self.device
        )
        torch.cat(batch.next_state, out=next_state_batch)
        action_batch = torch.tensor(
            batch.action, device=self.device
        ).reshape(self.batchsize, 1).float()
        reward_batch = torch.tensor(
            batch.reward, device=self.device
        ).reshape(self.batchsize, 1).float()

        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch.long())
        # metrics, q values average
        self.bag.append(torch.mean(self.policy_net.q_values))
        next_state_values = torch.zeros(self.batchsize, device=self.device)
        next_state_values = self.target_net(
            next_state_batch
        ).max(1)[0].detach().unsqueeze(1)
        expected_state_action_values = \
            next_state_values * self.gamma + reward_batch
        # loss
        loss = self.criterion(state_action_values.float(),
                              expected_state_action_values.float())
        loss.backward()
        # grad clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        # optimization
        self.optimizer.step()


class ExternalDQNAgent(Agent):
    """
    don't forget to set the agent actions with the set_actions method
    Same as DQNAgent, but the agent does not have its own DQN.
    """

    def __init__(self, params: DQNAgentParameters, actions):
        super(ExternalDQNAgent, self).__init__(params, actions)
        self.device = torch.device("cuda")
        self.action = 0
        self.action_index = 0
        self.d2d_tx = None

    def set_distance_to_bs(self, distance: float):
        self.distance_to_bs = distance

    def get_action(self, policy: ExternalDQNFramework, obs):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() > self.epsilon:
            # aux = torch.tensor([obs[0]], device=self.device)
            self.action_index = policy.policy_net(obs).max(1)[1].item()
            self.action = self.actions[self.action_index]
        else:
            self.action_index = torch.tensor(
                np.random.choice([i for i in range(len(self.actions))])
            ).item()
            # self.action_index = torch.tensor(
            #     self.action_index, device=self.device
            # )
            self.action = self.actions[self.action_index]
        return self.action

    def act(self, framework: ExternalDQNFramework, obs: torch.Tensor):
        return framework.policy_net(obs)

    def act_rainbow(self, framework: RainbowFramework, obs: torch.Tensor):
        return framework.policy_net.qvals(obs)

    def get_action_rainbow(self, policy: RainbowFramework, obs):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() > self.epsilon:
            # aux = torch.tensor([obs[0]], device=self.device)
            self.action_index = policy.policy_net.qvals(obs).max(1)[1].item()
            self.action = self.actions[self.action_index]
        else:
            self.action_index = torch.tensor(
                np.random.choice([i for i in range(len(self.actions))])
            ).item()
            # self.action_index = torch.tensor(
            #     self.action_index, device=self.device
            # )
            self.action = self.actions[self.action_index]
        return self.action

    def set_action(self, action_index: torch.Tensor, action: torch.Tensor):
        self.action_index = action_index
        self.action = action

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_d2d_tx(self, d2d_tx: d2d_user):
        self.d2d_tx = d2d_tx


class CentralDQNAgent(Agent):
    """
    don't forget to set the agent actions with the set_actions method
    Same as DQNAgent, but the agent does not have its own DQN.
    """

    def __init__(self, params: DQNAgentParameters, actions, n_agents):
        super(CentralDQNAgent, self).__init__(params, actions)
        self.device = torch.device("cuda")
        self.action_index = 0
        self.n_agents = n_agents
        self.actions_range = range(len(actions) ** n_agents)

    def get_action(self, policy: ExternalDQNFramework, obs):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() > self.epsilon:
            # aux = torch.tensor([obs[0]], device=self.device)
            self.action_index = policy.policy_net(obs).max(1)[1]
        else:
            self.action_index = torch.tensor(
                np.random.choice(list(self.actions_range))
            ).cpu()
            self.action_index = torch.tensor(
                self.action_index, device=self.device
            )
        return self.action_index

    def act(self, framework: ExternalDQNFramework, obs: torch.Tensor):
        return framework.policy_net(obs).max(1)[1].item()

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
