
import matplotlib.pyplot as plt
import numpy as np

sinrs = np.linspace(0, 20, 1000)
sinr_min = 10


def r3(sinr_m, sinr_min):
    g3 = 3
    if sinr_m >= sinr_min:
        r3 = g3*(2-2**(-sinr_m+sinr_min+1))
    else:
        r3 = -2*g3+g3**(sinr_m/5)
    return r3


rewards = [r3(i, sinr_min) for i in sinrs]


plt.figure()
plt.plot(sinrs, rewards)
plt.show()
