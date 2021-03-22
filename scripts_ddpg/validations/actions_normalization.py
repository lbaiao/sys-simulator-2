# validate if the agent's actions are normalized between 0 and 1
# from random import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from sys_simulator.ddpg import DDPGActor
from sys_simulator.ddpg.agent import Agent
from sys_simulator.ddpg.framework import Framework
# from sys_simulator.general import scale_tanh

N_SAMPLES = int(1e6)
N_BINS = int(1e2)
OBS_SIZE = 30
ACTION_SIZE = 3
tanh_min = -1
tanh_max = 1
a_min = 1e-9
a_max = 1

# Scaling validation. Do not delete.
# diff = tanh_max - tanh_min
# inputs = [random()*diff - diff/2 for _ in range(N_SAMPLES)]
# scaled = scale_tanh(np.array(inputs), a_min, a_max)
# plt.figure()
# plt.hist(scaled, density=True, bins=N_BINS)

# actor NN validation
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
actor = DDPGActor(OBS_SIZE, ACTION_SIZE, 128, 2, 3e-3).to(torch_device)
ac_inputs = np.random.random((N_SAMPLES, OBS_SIZE))
ac_outputs = actor(torch.FloatTensor(
    ac_inputs).to(torch_device)).detach().cpu().numpy()
ac_outputs = ac_outputs.reshape((N_SAMPLES, ACTION_SIZE))
plt.figure()
for i in range(ACTION_SIZE):
    plt.hist(ac_outputs[:, i], density=True, bins=N_BINS)
plt.savefig("figs/env_norm_actor_val.jpg")

# agent validation
framework = Framework('standard', 100, 10, OBS_SIZE, ACTION_SIZE, 128,
                      2, 3e-3, 3e-3, 64, .99, .999, torch_device)
framework.actor.eval()
framework.critic.eval()
agent = Agent(a_min, a_max, 'ou', torch_device)
ac_inputs = np.random.random((N_SAMPLES, OBS_SIZE))
ac_outputs = agent.act(ac_inputs, framework, False).detach().cpu().numpy()
ac_outputs = ac_outputs.reshape((N_SAMPLES, ACTION_SIZE))
plt.figure()
for i in range(ACTION_SIZE):
    plt.hist(ac_outputs[:, i], density=True, bins=N_BINS)
plt.savefig("figs/env_norm_agent_val.jpg")
print('done')
