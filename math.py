import torch
import math
import numpy as np
import scipy
import utils_torch

from utils_torch.attrs import GetAttrs

def NpArrayStatistics(data, verbose=False, ReturnType="PyObj"):
    DataStats = {
        "Min": np.nanmin(data),
        "Max": np.nanmax(data),
        "Mean": np.nanmean(data),
        "Std": np.nanstd(data),
        "Var": np.nanvar(data)
    }
    return utils_torch.Dict2GivenType(DataStats, ReturnType)

NpStatistics = NpArrayStatistics

def ReplaceNaNOrInfWithZeroNp(data):
    data[~np.isfinite(data)] = 0.0
    return data
ReplaceNaNOrInfWithZero = ReplaceNaNOrInfWithZeroNp

def IsAllNaNOrInf(data):
    return (np.isnan(data) | np.isinf(data)).all()

def RemoveNaNOrInf(data):
    # data: 1D np.ndarray.
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


def Norm2Mean0Std1Torch(data, axis=None, StdThreshold=1.0e-9):
    std = torch.std(data, dim=axis, keepdim=True)
    mean = torch.mean(data, dim=axis, keepdim=True)
    # if std < StdThreshold:
    #     utils_torch.AddWarning("ToMean0Std1Np: StandardDeviation==0.0")
    #     return data - mean
    # else:
    # To Be Implemented: Deal with std==0.0
    return (data - mean) / std

def Norm2Mean0Std1Np(data, axis=None, StdThreshold=1.0e-9):
    std = np.std(data, axis=axis, keepdims=True)
    mean = np.mean(data, axis=axis, keepdims=True)
    return (data - mean) / std

def Norm2Sum1(data, axis=None):
    # data: np.ndarray. Non-negative.
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

def CalculatePearsonCoefficientMatrix(dataA, dataB):
    # dataA: Design matrix of shape [SampleNum, FeatureNumA]
    # dataB: Design matrix of shape [SampleNum, FeatureNumB]
    FeatureNumA = dataA.shape[1]
    FeatureNumB = dataB.shape[1]
    SampleNum = dataA.shape[0]

    Location = utils_torch.GetGlobalParam().system.TensorLocation

    dataAGPU = utils_torch.ToTorchTensor(dataA).to(Location)
    dataBGPU = utils_torch.ToTorchTensor(dataB).to(Location)

    dataANormed = Norm2Mean0Std1Torch(dataAGPU, axis=0)
    dataBNormed = Norm2Mean0Std1Torch(dataBGPU, axis=0)

    CorrelationMatrix = torch.mm(dataANormed.permute(1, 0), dataBNormed) / SampleNum
    CorrelationMatrix = utils_torch.TorchTensor2NpArray(CorrelationMatrix)
    return CorrelationMatrix

def CalculateBinnedMeanAndStd(
        Xs, Ys, BinNum=30, BinMethod="Overlap", ReturnType="PyObj",
        Range="MinMax",
    ):
    if Range in ["MinMax"]:
        XMin, XMax = np.nanmin(Xs), np.nanmax(Xs)
    else:
        raise Exception(Range)

    if BinMethod in ["Overlap"]:
        BinNumTotal = 2 * BinNum - 1
        BinCenters = np.linspace(XMin, XMax, BinNumTotal + 2)[1:-1]

        BinWidth = (XMax - XMin) / BinNum
        Bins1 = np.linspace(XMin, XMax, BinNum + 1)
        
        BinMeans1, BinXs, BinNumber = scipy.stats.binned_statistic(
            Xs, Ys, statistic='mean', bins=Bins1)
        BinStds1, BinXs, BinNumber = scipy.stats.binned_statistic(
            Xs, Ys, statistic='std', bins=Bins1)
        BinCount1, BinXs, BinNumber = scipy.stats.binned_statistic(
            Xs, Ys, statistic='count', bins=Bins1)

        Bins2 = Bins1 + BinWidth / 2.0
        Bins2 = Bins2[:-1]

        BinMeans2, BinXs, BinNumber = scipy.stats.binned_statistic(
            Xs, Ys, statistic='mean', bins=Bins1)
        BinStds2, BinXs, BinNumber = scipy.stats.binned_statistic(
            Xs, Ys, statistic='std', bins=Bins1)
        BinCount2, BinXs, BinNumber = scipy.stats.binned_statistic(
            Xs, Ys, statistic='count', bins=Bins1)

        BinMeans, BinStds, BinCount = [], [], []
        for BinIndex in range(BinNum-1):
            BinMeans.append(BinMeans1[BinIndex])
            BinMeans.append(BinMeans2[BinIndex])
            BinStds.append(BinStds1[BinIndex])
            BinStds.append(BinStds2[BinIndex])
            BinCount.append(BinCount1[BinIndex])
            BinCount.append(BinCount2[BinIndex])
        
        BinMeans.append(BinMeans1[BinNum - 1])
        BinStds.append(BinStds1[BinNum - 1])
        BinCount.append(BinCount1[BinNum - 1])

        BinStats = {
            "Mean": utils_torch.List2NpArray(BinMeans),
            "Std": utils_torch.List2NpArray(BinStds),
            "Num": utils_torch.List2NpArray(BinCount),
            "BinCenters": BinCenters,
        }
        return utils_torch.Dict2GivenType(BinStats, ReturnType)
    else:
        raise Exception(BinMethod)

import sklearn
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

class Analyzer_PCA:
    def __init__(self, dim_num=3):
        self.pca = PCA(n_components=dim_num)
    def fit(self, data): #data:[sample_num, feature_size]
        self.pca.fit(data)
    def visualize_traj(self, trajs): #data:[traj_num][traj_length, N_num]
        fig, ax = utils_torch.plot.CreateFigurePlt()
        plt.title("Neural Trajectories")
        ax = fig.gca(projection='3d')
        mpl.rcParams['legend.fontsize'] = 10
        for traj in trajs:
            traj_trans = self.pca.transform(traj) #[dim_num, traj_length]
            ax.plot(traj_trans[:,0], traj_trans[:, 1], traj_trans[:, 2], label='parametric curve')
        plt.show()
        plt.savefig("./trajs_PCA3d.png")

def PCA(data):
    data = utils_torch.ToNpArray(data)

    return

PrincipalComponentAnalysis = PCA

