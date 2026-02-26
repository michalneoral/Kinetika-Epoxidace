import numpy as np

def make_immutable_array(arr):
    arr.setflags(write=False)
    return arr


K_MIN = 0.0
K_MAX = 1.0

