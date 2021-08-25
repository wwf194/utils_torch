import math
import cmath
import numpy as np

def polar2xy(r, theta):
    return r * math.cos(theta), r * math.sin(theta)
def xy2polar(x, y):
    return cmath.polar(complex(x, y))

def xy2polar_np(points): # [point_num, (x, y)]
    return np.arctan2(points[:, 1], points[:, 0])

def CosineSimilarityNumpy(vecA, vecB):
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    #normA_ = np.sum(vecA ** 2) ** 0.5
    #normB_ = np.sum(vecB ** 2) ** 0.5
    consine_similarity = np.dot(vecA.T, vecB) / (normA * normB)
    return consine_similarity