from sys_simulator.general import db_to_power, power_to_db
import numpy as np
# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

# sinr dB
min_mue_sinr = 6
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
        reward = db_to_power(mue_sinr) - \
            db_to_power(min_mue_sinr)
    return reward


def calculate_continuous_reward(
    mue_sinr, d2d_sinrs, min_mue_sinr, gamma1=2.0, gamma2=1.0
):    
    if mue_sinr <= min_mue_sinr:
        reward = gamma1 * (mue_sinr - min_mue_sinr)
    else:
        coef2 = gamma2 * 10
        if d2d_sinrs < coef2:
            reward = coef2 ** (d2d_sinrs/10)
            # reward = db_to_power(d2d_sinrs)
            # reward = 1e-9 if reward == 0 else reward
        else:
            reward = d2d_sinrs
    return reward


plt.figure()
rewards1 = []
rewards2 = []
rewards3 = []
# continuous reward
for m, d in zip(mue_sinr1, d2d_sinrs1):
    rewards1.append(calculate_continuous_reward(m, d, min_mue_sinr))
plt.plot(mue_sinr1, rewards1, label=r'$S_M < \tau_0$, $R=(S_M-\tau_0)_{dB}$')
for m, d in zip(mue_sinr2, d2d_sinrs2):
    rewards2.append(calculate_continuous_reward(m, d, min_mue_sinr))
plt.plot(mue_sinr2, rewards2,
         label=r'$S_M \geq \tau_0$, $S_D \leq 10$dB, $R=S_D$')
for m, d in zip(mue_sinr3, d2d_sinrs3):
    rewards3.append(calculate_continuous_reward(m, d, min_mue_sinr))
plt.plot(mue_sinr3, rewards3,
         label=r'$S_M \geq \tau_0$, $S_D > 10$dB, $R=S_{D_{dB}}$')
# other reward
# for m, d in zip(mue_sinr1, d2d_sinrs1):
    # rewards1.append(reward(m, d, min_mue_sinr))
# plt.plot(mue_sinr1, rewards1, label=r'$S_M < \tau_0$, $R=S_M-\tau_0$')
# for m, d in zip(mue_sinr2, d2d_sinrs2):
    # rewards2.append(reward(m, d, min_mue_sinr))
# plt.plot(mue_sinr2, rewards2,
         # label=r'$S_M \geq \tau_0$, $R=S_{D_{dB}}$')
# for m, d in zip(mue_sinr3, d2d_sinrs3):
    # rewards3.append(reward(m, d, min_mue_sinr))
# plt.plot(mue_sinr3, rewards3,
#          label=r'$S_M \geq \tau_0$, $R=S_{D_{dB}}$')
# plot
plt.xlabel('MUE SINR [dB]')
plt.ylabel('reward')
plt.legend()
plt.show()
