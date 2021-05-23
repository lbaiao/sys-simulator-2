from sys_simulator.general import db_to_power, power_to_db
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform

tau0 = 6
m0 = -60
m_min = -90
size = 10000


def scaling(values):
    values = np.clip(values, m_min, m0)
    return (values - m_min)/(m0 - m_min)


def continuous_reward2(mue_sinr, d2d_sinrs):
    if mue_sinr < tau0:
        return mue_sinr - tau0 - 10
    mask = d2d_sinrs < m0
    if np.sum(mask) > 0:
        return scaling(d2d_sinrs[mask].mean()) * 10 - 10
    r = np.sum(db_to_power(d2d_sinrs))
    if r <= 10:
        return r
    return power_to_db(r)


mue_sinrs = uniform(-40, 40, size)
d2d_sinrs = uniform(-100, 20, (size, 2))

rewards = []
r_m = []
for m, d in zip(mue_sinrs, d2d_sinrs):
    rew = continuous_reward2(m, d)
    rewards.append(rew)
    r_m.append((rew, m))
rewards.sort()
r_m.sort(key=lambda x: x[0])
r_m = np.array(r_m)

scalings = scaling(np.linspace(-100, -60, size)) * 10

m1 = sorted(mue_sinrs)
r1 = np.array(m1) - tau0 - 10


plt.figure()
# plt.plot(r_m[:, 1], r_m[:, 0])
r_mx = r_m[r_m[:, 1] < tau0]
plt.plot(r_mx[:, 1], label='mue sinrs')
plt.plot(r_mx[:, 0], label='rewards')
plt.plot(r_m[:, 1], label='mue sinrs')
plt.plot(r_m[:, 0], label='rewards')
plt.legend()
plt.yticks([-20, -10, 0, 10, 20])

# plt.figure()
# plt.plot(scalings)
# plt.yticks([-10, 0])

plt.figure()
plt.plot(m1, r1)
plt.yticks([-10, 0, tau0])
plt.xticks([-10, 0, tau0, 10])



plt.show()
