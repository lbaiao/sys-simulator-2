# validate if the agent's actions are normalized between 0 and 1
from sys_simulator.general import scale_tanh
from random import random 
import matplotlib.pyplot as plt

N_SAMPLES = int(1e6)
N_BINS = int(1e2)
tanh_min = -1
tanh_max = 1
a_min = 1e-9
a_max = 1e9

diff = tanh_max - tanh_max
inputs = [random()*diff - diff/2 for _ in range(N_SAMPLES)]
plt.hist(inputs, density=True)
plt.show()
