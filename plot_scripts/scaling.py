import matplotlib.pyplot as plt
import numpy as np

from sys_simulator.general import db_to_power, power_to_db, scale_tanh


p_max = 40  # max tx power in dBm
p_max = p_max - 30
a_min = 0 + 1e-9
a_max = db_to_power(p_max - 10)

x = np.linspace(-1, 1, 1000)
y = scale_tanh(x, a_min, a_max)

plt.figure()
plt.plot(x, y)

y2 = power_to_db(y)
plt.figure()
plt.plot(x, y2)

plt.show()
