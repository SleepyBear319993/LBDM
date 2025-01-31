import math
import cupy as cp

class D2Q9:
    w = cp.array([4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0])
    cx = cp.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cy = cp.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    cs2 = 1.0/3.0
    cs = math.sqrt(cs2)