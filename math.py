from utils_torch.attrs import GetAttrs
import numpy as np

import utils_torch

def NpStatistics(data, verbose=False):
    return utils_torch.json.Dict2PyObj({
        "Min": np.min(data),
        "Max": np.max(data),
        "Mean": np.mean(data),
        "Std": np.std(data),
        "Var": np.var(data)
    })

def CreateNpArray(Shape, Value, DataType):
    return np.full(tuple(Shape), Value, dtype=DataType)

def SampleFromDistribution(param, Shape=None):
    if Shape is None:
        Shape = GetAttrs(param.Shape)
    if param.Type in ["Reyleigh"]:
        return SamplesFromReyleighDistribution(
            Mean = GetAttrs(param.Mean),
            Shape = Shape,
        )
    elif param.Type in ["Gaussian", "Gaussian1D"]:
        return SampleFromGaussianDistribution(
            Mean = GetAttrs(param.Mean),
            Std = GetAttrs(param.Std),
            Shape = Shape,
        )    
    else:
        raise Exception()

def SampleFromGaussianDistribution(Mean=0.0, Std=1.0, Shape=100):
    return np.random.normal(loc=Mean, scale=Std, size=utils_torch.parse.ParseShape(Shape))

def SamplesFromReyleighDistribution(Mean=1.0, Shape=100):
    # p(x) ~ x^2 / sigma^2 * exp( - x^2 / (2 * sigma^2))
    # E[X] = 1.253 * sigma
    # D[X] = 0.429 * sigma^2
    Shape = utils_torch.parse.ParseShape(Shape)
    return np.random.rayleigh(Mean / 1.253, Shape)

def CosineSimilarityNp(vecA, vecB):
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    #normA_ = np.sum(vecA ** 2) ** 0.5
    #normB_ = np.sum(vecB ** 2) ** 0.5
    CosineSimilarity = np.dot(vecA.T, vecB) / (normA * normB)
    return CosineSimilarity

def Vectors2Directions(Vectors):
    Directions = []
    for Vector in Vectors:
        R, Direction = utils_torch.geometry2D.XY2Polar(*Vector)
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
