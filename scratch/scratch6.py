# channel loss curves
from sys_simulator.channels import BANChannel
from sys_simulator.pathloss import pathloss_bs_users, pathloss_users
import numpy as np
import matplotlib.pyplot as plt

# distances, in meters
distances = np.linspace(1e-9, 1000, 10000)
ban_office = BANChannel()
ban_ferry = BANChannel(env='ferry')
user_pathloss = pathloss_users(distances/1000)
bs_pathloss = pathloss_bs_users(distances/1000)
office_pathloss = ban_office.step(distances)[1]
ferry_pathloss = ban_ferry.step(distances)[1]
# plots
plt.figure()
plt.semilogy(distances, user_pathloss, label='to user')
plt.semilogy(distances, bs_pathloss, label='to BS')
plt.semilogy(distances, office_pathloss, label='BAN office')
plt.semilogy(distances, ferry_pathloss, label='BAN ferry')
plt.legend()
plt.xlabel('Distance [m]')
plt.ylabel('Channel Pathloss')
plt.show()
