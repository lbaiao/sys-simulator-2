import matplotlib.pyplot as plt
from sys_simulator.devices.motion_models import MotionModel
import numpy as np

m_uni = MotionModel('gauss_pedestrian')
m_walk = MotionModel('walking_pedestrian')
m_stopped = MotionModel('no_movement')

dt = 1e-3
STEPS = 10000
steps = np.linspace(0, STEPS, STEPS)

p1 = (0, 0, 1.5)
d1 = 0
p2 = (0, 0, 1.5)
d2 = 0
p3 = (0, 0, 1.5)
d3 = 0
p1_hist = [p1]
p2_hist = [p2]
p3_hist = [p3]

for t in steps:
    p1, d1 = m_uni.step(p1, d1, dt)
    p2, d2 = m_walk.step(p2, d2, dt)
    p3, d3 = m_stopped.step(p3, d3, dt)
    p1_hist.append(p1)
    p2_hist.append(p2)
    p3_hist.append(p3)

p1_x, p1_y, _ = zip(*p1_hist)
p2_x, p2_y, _ = zip(*p2_hist)
p3_x, p3_y, _ = zip(*p3_hist)

plt.figure()
plt.plot(p1_x, p1_y, label='Gauss pedestrian')
plt.plot(p2_x, p2_y, label='Walking pedestrian')
plt.plot(p3_x, p3_y, 'd', label='No movement')
plt.title(f't = {STEPS * dt} s')
plt.legend()
plt.savefig('figs/motion_models.png')
plt.show()
