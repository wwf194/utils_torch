import math
import cmath
import numpy as np

def polar2xy(r, theta):
    return r * math.cos(theta), r * math.sin(theta)
def xy2polar(x, y):
    return cmath.polar(complex(x, y))

def xy2polar_np(points): # [point_num, (x, y)]
    return np.arctan2(points[:, 1], points[:, 0])