import torch
from sys_simulator.a2c.a2c import ActorCritic


class Agent:
    def __init__(self):
        self.bag = list()
        self.action = 0
        self.device =\
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_d2d_tx_id(self, id: str):
        self.id = id

    def act(self, a2c: ActorCritic, obs: torch.TensorType):
        dist, value = a2c(obs)
        self.action = (dist.sample()*1e-5).item()

        return self.action, dist, value

    def get_action(self):
        return self.action

    def set_action(self, action):
        self.action = action
