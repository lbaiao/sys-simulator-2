from sys_simulator.devices.devices import db_to_power
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

# sinr linear
min_mue_sinr = db_to_power(6)
n_samples = 1000
mue_sinr1 = np.linspace(-10, min_mue_sinr, n_samples)
mue_sinr2 = np.linspace(min_mue_sinr, 10, n_samples)
mue_sinr3 = np.linspace(10, 20, n_samples)
d2d_sinrs1 = np.linspace(-10, 10, n_samples)
d2d_sinrs2 = np.linspace(-10, 10, n_samples)
d2d_sinrs3 = np.linspace(10, 20, n_samples)


def reward(mue_sinr, d2d_sinrs, min_mue_sinr):
    if mue_sinr > min_mue_sinr:
        reward = np.sum(db_to_power(d2d_sinrs))
        reward = 1e-9 if reward == 0 else reward
        reward = 0.2 * power_to_db(reward)
    else:
        reward = (mue_sinr - min_mue_sinr)**3
    return reward


plt.figure()
rewards1 = []
rewards2 = []
for m, d in zip(mue_sinr1, d2d_sinrs):
    rewards1.append(reward(m, d, min_mue_sinr))
plt.plot(mue_sinr1, rewards1, label=r'mue sinr $< \tau_0$')
for m, d in zip(mue_sinr2, d2d_sinrs):
    rewards2.append(reward(m, d, min_mue_sinr))
plt.plot(mue_sinr2, rewards2, label=r'mue sinr $\geq \tau_0$')
plt.xlabel('MUE SINR')
plt.ylabel('reward')
plt.legend()
plt.show()
