import numpy as np

def metric_mae(x, y):
    return np.mean(np.abs(x-y))