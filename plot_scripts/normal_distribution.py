import matplotlib.pyplot as plt
from numpy.random import normal

size = 10000
loc = 0
scale = 450

y = normal(loc, scale, size)

plt.figure()
plt.hist(y, bins=100, density=True)
plt.show()

