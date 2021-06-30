import random
import numpy as np


class SimpleMemory(object):
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.memory = []
        self.position = 0

    def push(self, item: object):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MinMaxMemory:
    def __init__(self):
        self.a_min = float('inf')
        self.a_max = float('-inf')

    def push(self, val: float):
        self.a_min = np.min([self.a_min, *val])
        self.a_max = np.max([self.a_max, *val])
