from sys_simulator.noises.decaying_gauss_noise import DecayingGaussNoise
import matplotlib.pyplot as plt
import numpy as np
from math import log

loc = 0
scale = 2
min_decay = .05
T = 30000
size = 2
N_SAMPLES = 3*T

steps = np.linspace(0, T, N_SAMPLES)
noise = DecayingGaussNoise(loc, scale, T, min_decay, size)

outs = []
scales = []
for t in steps:
    scales.append(noise.scale)
    outs.append(noise.step(t))
f, axs = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
for o in zip(*outs):
    axs[0].plot(steps, o)
axs[0].set_title('Noise')
axs[1].plot(steps, scales)
axs[1].set_title('Decay')
axs[0].set_xlabel('Step')
axs[1].set_xlabel('Step')
f.align_xlabels()
plt.savefig('figs/noise_decay.png')
plt.show()

