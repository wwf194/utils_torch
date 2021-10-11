import torch
import math
import numpy as np
import utils_torch

from utils_torch.attrs import GetAttrs

def NpArrayStatistics(data, verbose=False, ReturnType="PyObj"):
    statistics = {

        "Min": np.min(data),
        "Max": np.max(data),
        "Mean": np.mean(data),
        "Std": np.std(data),
        "Var": np.var(data)
    }
    if ReturnType in ["Dict"]:
        return statistics
    elif ReturnType in ["PyObj"]:
        return utils_torch.PyObj(statistics)
    else:
        raise Exception()

NpStatistics = NpArrayStatistics

def ReplaceNaNOrInfWithZeroNp(data):
    data[~np.isfinite(data)] = 0.0
    return data
ReplaceNaNOrInfWithZero = ReplaceNaNOrInfWithZeroNp

def IsAllNaNOrInf(data):
    return (np.isnan(data) | np.isinf(data)).all()

def RemoveNaNOrInf(data):
    # @param data: 1D np.ndarray.
    return data[np.isfinite(data)]

def TorchTensorStat(tensor, verbose=False, ReturnType="PyObj"):
    statistics = {
        "Min": torch.min(tensor).item(),
        "Max": torch.max(tensor).item(),
        "Mean": torch.mean(tensor).item(),
        "Std": torch.std(tensor).item(),
        "Var": torch.var(tensor).item()
    }
    if ReturnType in ["Dict"]:
        return statistics
    elif ReturnType in ["PyObj"]:
        return utils_torch.PyObj(statistics)
    else:
        raise Exception()

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

def SampleFromGaussianDistributionTorch(Mean=0.0, Std=1.0, Shape=100):
    data = SampleFromGaussianDistribution(Mean, Std, Shape)
    data = utils_torch.NpArray2Tensor(data)
    return data

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

def IsAcuteAnglesNp(AnglesA, AnglesB):
    return np.abs(Angles2StandardRangeNp(AnglesA, AnglesB)) < np.pi / 2

def ToMean0Std1Np(data, StdThreshold=1.0e-9):
    std = np.std(data, keepdims=True)
    mean = np.mean(data, keepdims=True)
    if std < StdThreshold:
        utils_torch.AddWarning("ToMean0Std1Np: StandardDeviation==0.0")
        return data - mean
    else:
       return (data - mean) / std

ToMean0Std1 = ToMean0Std1Np

def Norm2GivenMeanStdNp(data, Mean, Std, StdThreshold=1.0e-9):
    data = ToMean0Std1Np(data, StdThreshold)
    return data * Std + Mean

Norm2GivenMeanStd = Norm2GivenMeanStdNp

def ToMean0Std1Torch(data, axis=None, StdThreshold=1.0e-9):
    std = torch.std(data, dim=axis, keepdim=True)
    mean = torch.mean(data, dim=axis, keepdim=True)
    # if std < StdThreshold:
    #     utils_torch.AddWarning("ToMean0Std1Np: StandardDeviation==0.0")
    #     return data - mean
    # else:


    # To Be Implemented: Deal with std==0.0
    return (data - mean) / std

def Norm2Sum1(data, axis=None):
    # @param data: np.ndarray. Non-negative.
    data / np.sum(data, axis=axis, keepdims=True)

def Norm2Range01(data, axis=None):
    Min = np.min(data, axis=axis, keepdims=True)
    Max = np.max(data, axis=axis, keepdims=True)
    return (data - Min) / (Max - Min)

def Norm2Min0(data, axis=None):
    Min = np.min(data, axis=axis, keepdims=True)
    return data - Min

Norm2ProbDistribution = Norm2ProbabilityDistribution = Norm2Sum1

GaussianCoefficient = 1.0 / (2 * np.pi) ** 0.5

def GetGaussianProbDensityMethod(Mean, Std):
    # return Gaussian Probability Density Function
    CalculateExponent = lambda data: - 0.5 * ((data - Mean) / Std) ** 2
    Coefficient = GaussianCoefficient / Std
    ProbDensity = lambda data: Coefficient * np.exp(CalculateExponent(data))
    return ProbDensity

def GetGaussianCurveMethod(Amp, Mean, Std):
    # return Gaussian Curve Function.
    CalculateExponent = lambda data: - 0.5 * ((data - Mean) / Std) ** 2
    GaussianCurve = lambda data: Amp * np.exp(CalculateExponent(data))
    return GaussianCurve

def GaussianProbDensity(data, Mean, Std):
    Exponent = - 0.5 * ((data - Mean) / Std) ** 2
    return GaussianCoefficient / Std * np.exp(Exponent)

def GaussianCurveValue(data, Amp, Mean, Std):
    Exponent = - 0.5 * ((data - Mean) / Std) ** 2
    return Amp * np.exp(Exponent)

def Float2BaseAndExponent(Float, Base=10.0):
    Exponent = math.floor(math.log(Float, Base))
    Coefficient = Float / 10.0 ** Exponent
    return Coefficient, Exponent

Float2BaseExp = Float2BaseAndExponent

def Floats2BaseAndExponent(Floats, Base=10.0):
    Floats = utils_torch.ToNpArray(Floats)
    Exponent = np.ceil(np.log10(Floats, Base))
    Coefficient = Floats / 10.0 ** Exponent
    return Coefficient, Exponent

def CalculatePearsonCoefficient(dataA, dataB):
    # dataA: Design matrix of shape [SampleNum, FeatureNumA]
    # dataB: Design matrix of shape [SampleNum, FeatureNumB]
    FeatureNumA = dataA.shape[1]
    FeatureNumB = dataB.shape[1]
    SampleNum = dataA.shape[0]

    Location = utils_torch.GetArgsGlobal().system.TensorLocation

    dataAGPU = utils_torch.ToTorchTensor(dataA).to(Location)
    dataBGPU = utils_torch.ToTorchTensor(dataB).to(Location)

    dataANormed = ToMean0Std1Torch(dataAGPU, axis=0)
    dataBNormed = ToMean0Std1Torch(dataBGPU, axis=0)

    CorrelationMatrix = torch.mm(dataANormed.permute(1, 0), dataBNormed) / SampleNum
    CorrelationMatrix = utils_torch.TorchTensor2NpArray(CorrelationMatrix)
    return CorrelationMatrix