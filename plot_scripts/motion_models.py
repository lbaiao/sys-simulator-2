import matplotlib.pyplot as plt
from sys_simulator.devices.motion_models import MotionModel
import numpy as np

m_uni = MotionModel('gauss_pedestrian')
m_walk = MotionModel('walking_pedestrian')

dt = 1e-3
STEPS = 10000
steps = np.linspace(0, STEPS, STEPS)

p1 = (0, 0)
d1 = 0
p2 = (0, 0)
d2 = 0
p1_hist = [p1]
p2_hist = [p2]

for t in steps:
    p1, d1 = m_uni.step(p1, d1, dt)
    p2, d2 = m_walk.step(p2, d2, dt)
    p1_hist.append(p1)
    p2_hist.append(p2)

p1_x, p1_y = zip(*p1_hist)
p2_x, p2_y = zip(*p2_hist)

plt.figure()
plt.plot(p1_x, p1_y, label='Uniform pedestrian')
plt.plot(p2_x, p2_y, label='Walking pedestrian')
plt.title(f't = {STEPS * dt} s')
plt.legend()
plt.savefig('figs/motion_models.png')
plt.show()
