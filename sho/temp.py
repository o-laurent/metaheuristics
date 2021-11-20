import numpy as np


def proportional(i: int, max_iter: int):
    return max_iter/(i+1)


def square(i: int, max_iter: int):
    return max_iter/(i*i)


def logarithmic(i: int, max_iter: int, base:int=np.math.e):
    return max_iter/(np.math.log(i+1, base)+1)


def root(i: int, max_iter: int, power:float=1/2):
    # Does not work
    return max_iter/(np.power(i, power)+1)
