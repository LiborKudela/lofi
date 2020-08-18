from numpy import random, inf

class uniform():
    def __init__(self, low=-0.1, high=0.1):
        self.low = low
        self.high = high

    def initialize(self,M):
        M.p[:] = random.uniform(self.low, self.high, M.m)
        M.y = inf

class normal():
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def initialize(self,M):
        M.p[:] = random.normal(self.mean, self.std, M.m)
        M.y = inf

class om_override_file():
    def __init__(self, path):
        self.path = path

    def initialize(self, M):
        M.p[:] = self.p
