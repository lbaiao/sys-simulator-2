# channel loss curves
from sys_simulator.channels import BANChannel, UrbanMacroLOSWinnerChannel
from sys_simulator.pathloss import pathloss_bs_users, pathloss_users
import numpy as np
import matplotlib.pyplot as plt

# distances, in meters
distances = np.linspace(10, 1000, 10000)
ban_office = BANChannel()
ban_ferry = BANChannel(env='ferry')
user_pathloss = pathloss_users(distances/1000)
user_pathloss = 10 * np.log10(user_pathloss)
bs_pathloss = pathloss_bs_users(distances/1000)
bs_pathloss = 10 * np.log10(bs_pathloss)
office_pathloss = ban_office.step(distances)
ferry_pathloss = ban_ferry.step(distances)
winner_pathloss = [UrbanMacroLOSWinnerChannel(
    d, 15, 1.5, 2.4
) for d in distances]
# plots
plt.figure()
plt.plot(distances, user_pathloss, label='to user')
plt.plot(distances, bs_pathloss, label='to BS')
plt.plot(distances, office_pathloss, label='BAN office')
plt.plot(distances, ferry_pathloss, label='BAN ferry')
plt.plot(distances, winner_pathloss, label='WINNER II')
plt.legend()
plt.xlabel('Distance [m]')
plt.ylabel('Channel Pathloss [dB]')
# plt.ylim([-100, 100])
plt.show()
