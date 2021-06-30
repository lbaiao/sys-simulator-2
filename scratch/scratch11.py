# channel loss curves
from sys_simulator.channels import UrbanMacroLOSWinnerChannel
import matplotlib.pyplot as plt
import numpy as np

DISTANCE = 300
RND = True
ITERATIONS = int(1E5)
channel = UrbanMacroLOSWinnerChannel(RND, 25, 1.5, 2.4)
# pathlosses = [channel.step(DISTANCE) for _ in range(ITERATIONS)]
# pathlosses = [np.random.normal(0, 1) for _ in range(ITERATIONS)]
# pathlosses = np.random.lognormal(0, 2, ITERATIONS)
pathlosses = np.random.normal(0, 6, ITERATIONS)
# plots
# plt.plot(pathlosses)
plt.hist(pathlosses, bins=1000, density=True)
# plt.xlabel('Iteration')
# plt.ylabel('Channel Pathloss [dB]')
# plt.ylim([-100, 100])
plt.show()
