import math
import cmath
import numpy as np

def polar2xy(Radius, Direction):
    # Direction: [-pi, pi)
    return Radius * math.cos(Direction), Radius * math.sin(Direction)
def xy2polar(x, y):
    return cmath.polar(complex(x, y))

def xy2polarNp(PointsNp): # [PointNum, (x, y)]
    return np.arctan2(PointsNp[:, 1], PointsNp[:, 0])

def Vertices2VertexPairs(Vertices, close=True):
    VertexNum = len(Vertices)
    VertexPairs = []
    if close:
        for Index in VertexNum:
            VertexPairs.append(Vertices[Index], Vertices[(Index + 1) % VertexNum])
    return

def Vertices2Vectors(Vertices, close=True): 
    return Vertices2EdgesNp(np.array(Vertices, dtype=np.float32), close=close).tolist()

def Vertices2EdgesNp(VerticesNp, close=True):
    if close:
        VerticesNp = np.concatenate((VerticesNp, VerticesNp[0,:][np.newaxis, :]), axis=0)
    VectorsNp = np.diff(VerticesNp, axis=0)
    return VectorsNp

def VertiexPairs2Vectors(VertexPairs):
    Vectors = []
    for VertexPair in VertexPairs:
        Vectors.append(VertexPair2Vector(VertexPair))
    return Vectors

def VertexPair2Vector(VertexPair):
    return [[VertexPair[0][1] - VertexPair[0][0]], VertexPair[1][1] - VertexPair[1][0]]

def VertexPair2VectorNp(VertexPairNp): # ((x0, y0), (x1, y1))
    return np.diff(VertexPairNp, axis=1)

def Vectors2Norms(Vectors):
    return Vectors2NormsNp(np.array(Vectors, dtype=np.float32)).tolist()

def Vectors2NormsNp(VectorsNp):  # Calculate Norm Vectors Pointing From Inside To Outside Of Polygon
    #VectorNum = len(Vectors)  
    #Vectors = np.array(Vectors, dtype=np.float32)
    VectorNum = VectorsNp.shape[0]
    VectorsNorm = np.zeros([VectorNum, 2])
    # (a, b) is vertical to (b, -a)
    VectorsNorm[:, 0] = VectorsNp[:, 1]
    VectorsNorm[:, 1] = - VectorsNp[:, 0]
    # Normalize To Unit Length
    VectorsNorm = VectorsNorm / (np.linalg.norm(VectorNorms, axis=1, keepdims=True))
    return VectorsNorm

def Vectors2DirectionsNp(Vectors):
    Directions = []
    Directions = xy2polarNp(Vectors)
    return Directions

def Vectors2Directions(Vectors):
    Directions = []
    for Vector in Vectors:
        Directions.append(xy2polar(Vector))
    return Directions

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
        R, Direction = xy2polar(*Vector)
        Directions.append(Direction)    
    return Directions

def Vectors2NormsNp(VectorsNp, axis=-1):
    return np.linalg.norm(VectorsNp, axis=axis)