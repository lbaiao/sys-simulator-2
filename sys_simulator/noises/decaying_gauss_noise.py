from numpy.random import normal
import numpy as np
from math import exp, log


class DecayingGaussNoise:
    def __init__(self, loc: float, scale: float, T: float, min_decay: float, size: int):
        self.loc = loc
        self.original_scale = scale
        self.scale = scale
        self.alpha = -log(min_decay/scale)
        self.T = T
        self.min_decay = min_decay
        self.size = size

    def step(self, t: float):
        scale = self.original_scale*exp(-self.alpha*t/self.T)
        scale = np.max([scale, self.min_decay])
        self.scale = scale
        noise = normal(loc=self.loc, scale=scale, size=self.size)
        return noise

    def reset(self):
        self.scale = self.original_scale
