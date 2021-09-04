import math
import cmath
import numpy as np

import numpy as np

def CreateArray(Shape, Value, DataType):
    return np.full(tuple(Shape), Value, dtype=DataType)


def CosineSimilarityNp(vecA, vecB):
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    #normA_ = np.sum(vecA ** 2) ** 0.5
    #normB_ = np.sum(vecB ** 2) ** 0.5
    consine_similarity = np.dot(vecA.T, vecB) / (normA * normB)
    return consine_similarity

def Vectors2Directions(Vectors):
    Directions = []
    for Vector in Vectors:
        R, Direction = XY2Polar(*Vector)
        Directions.append(Direction)    
    return Directions

def Vector2Norm(VectorNp):
    return np.linalg.norm(VectorNp)

def Vectors2NormsNp(VectorsNp): # VectorsNp: [VectorNum, VectorSize]
    return np.linalg.norm(VectorsNp, axis=-1)

def Angles2StandardRangeNp(Angles):
    return np.mod(Angles, np.pi * 2) - np.pi

def isAcuteAnglesNp(AnglesA, AnglesB):
    return np.abs(Angles2StandardRangeNp(AnglesA, AnglesB)) < np.pi / 2

isAcuteAngles = isAcuteAnglesNp

def StartEndPoints2VectorsNp(PointsStart, PointsEnd):
    return PointsStart - PointsEnd

def StartEndPoints2VectorsDirectionNp(PointsStart, PointsEnd):
    Vectors = StartEndPoints2VectorsNp(PointsStart, PointsEnd)
    return Vectors2DirectionsNp(Vectors)

def StartEndPoints2VectorsNormNp(PointsStart, PointsEnd):
    Vectors = StartEndPoints2VectorsNp(PointsStart, PointsEnd)
    return Vectors2NormsNp(Vectors)

def StartEndPoints2VectorsNormDirectionNp(PointsStart, PointsEnd):
    Vectors = StartEndPoints2VectorsNp(PointsStart, PointsEnd)
    return Vectors2NormsDirectionsNp(Vectors)