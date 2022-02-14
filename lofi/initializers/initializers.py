import numpy as np

def constant(value=0.0, shape=None):
    return np.full(shape, value)

def normal(mean=0.0, std=1.0, shape=None):
    return np.random.normal(loc=mean, scale=std, size=shape)

def uniform(low=0.0, up=1.0, shape=None):
    return np.uniform(low=lb, high=up, size=shape)
