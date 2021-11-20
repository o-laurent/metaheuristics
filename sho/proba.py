import numpy as np


def exponential(val: float, best_val: float, temperature: float):
    return np.math.exp((val - best_val)/temperature)
