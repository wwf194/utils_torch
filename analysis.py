import torch
import numpy as np
import scipy
import matplotlib as mpl
from matplotlib import pyplot as plt

import utils_torch

def AnalyzeTimeVaryingActivitiesEpochBatch(Logs, PlotIndex=0, EpochIndex=None, BatchIndex=None, SaveDir=None):
    if SaveDir is None:
        SaveDir = utils_torch.GetMainSaveDir() + "NeuronActivity-Plot/"
    PlotIndex = 0
    for name, activity in Logs.items():
        EpochIndex = activity["Epoch"]
        BatchIndex = activity["Batch"]
        activity = activity["Value"]
        _name = "%s-Epoch%d-Batch%d-No%d"%(name, EpochIndex, BatchIndex, PlotIndex)
        utils_torch.analysis.AnalyzeTimeVaryingActivity(
            activity,
            PlotIndex=PlotIndex,
            Name=_name, 
            SavePath=SaveDir + "%s/%s.svg"%(name, _name),
        )

def AnalyzeTimeVaryingActivity(activity, PlotIndex, Name=None, SavePath=None):
    utils_torch.plot.PlotActivityAndDistributionAlongTime(
        axes=None,
        activity=activity,
        activityPlot=activity[PlotIndex],
        Title=Name,
        Save=True,
        SavePath=SavePath,
    )

def AnalyzeWeightsEpochBatch(Logs, EpochIndex=None, BatchIndex=None, PlotIndex=0, SaveDir=None):
    if SaveDir is None:
        SaveDir = utils_torch.GetMainSaveDir() + "Weight-Plot/"
    PlotIndex = 0
    for Name, Weights in Logs.items():
        # EpochIndex = Weights[0]
        # BatchIndex = Weights[1]
        # weights = Weights[2]
        EpochIndex = Weights["Epoch"]
        BatchIndex = Weights["Batch"]
        weights = Weights["Value"]
  
        for name, weight in weights.items():
            _name = "%s-Epoch%d-Batch%d-No%d"%(name, EpochIndex, BatchIndex, PlotIndex)
            utils_torch.analysis.AnalyzeWeight(
                weight,
                Name=_name, 
                SavePath=SaveDir + "%s/%s.svg"%(name, _name),
            )

def AnalyzeWeight(weight, Name, SavePath=None):
    utils_torch.plot.PlotWeightAndDistribution(
        axes=None,
        weight=weight,
        Name=Name,
        SavePath=SavePath,
    )
    return

def AnalyzeWeightStatAlongTrainingEpochBatch(Logs, SaveDir=None, **kw):
    if SaveDir is None:
        SaveDir = utils_torch.GetMainSaveDir() + "Weight-Stat/"
    for Name, Log in Logs.items(): # Each Log is statistics of a weight along training process.
        assert isinstance(Log, dict)
        EpochIndices = Log["Epoch"]
        BatchIndices = Log["Batch"]
        EpochsFloat = utils_torch.train.EpochBatchIndices2EpochsFloat(
            EpochIndices, BatchIndices, BatchNum = kw["BatchNum"]
        )
        fig, ax = utils_torch.plot.CreateFigurePlt()
        utils_torch.plot.PlotMeanAndStdCurve(
            ax, Xs=EpochsFloat,
            Mean=Log["Mean"], Std=Log["Std"], 
            Title="%s - Epoch"%Name, XLabel="Epoch", YLabel=Name,
        )
        plt.tight_layout()
        utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "%s~Epoch.svg"%Name)
        utils_torch.files.Table2TextFileDict(Log, SavePath=SaveDir + "%s~Epoch.txt"%Name)
    return

def PlotLogDictStatistics(self, Name, Log, SaveDir=None):
    utils_torch.EnsureDir(SaveDir)
    Epochs = self.GetEpochsFloatFromLogDict(Log)
    fig, ax = plt.subplots()

def AnalyzeLossEpochBatch(Logs, SaveDir=None, **kw):
    if SaveDir is None:
        SaveDir = utils_torch.GetMainSaveDir() + "Loss/"
    EpochsFloat = utils_torch.log.ListLog2EpochsFloat(Logs, BatchNum=kw["BatchNum"])
    LossDict = {}
    for Name, Log in Logs.items(): # Each Log is statistics of a weight along training process.
        #assert isinstance(Log, list)
        Loss = Log["Value"]
        LossDict[Name] = Loss
    fig, ax = utils_torch.plot.CreateFigurePlt()
    utils_torch.plot.PlotMultiLineChart(
        ax, Xs=EpochsFloat, YsDict=LossDict,
        XTicks="Float", YTicks="Float",
        Title="Loss - Epoch", XLabel="Epoch", YLabel="Loss",
    )

    plt.tight_layout()
    utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "Loss~Epoch.svg")
    utils_torch.files.Table2TextFileDict(LossDict, SavePath=SaveDir + "Loss~Epoch.txt")
    return

def PlotResponseSimilarityAndWeightUpdateCorrelation(CorrelationMatrix, Weight, SaveDir, SaveName):
    fig, axes = utils_torch.plot.CreateFigurePlt(2, Size="Medium")
    
    CorrelationMatrixFlat = utils_torch.EnsureFlat(CorrelationMatrix)
    WeightFlat = utils_torch.EnsureFlat(Weight)
    
    ax = utils_torch.plot.GetAx(axes, 0)
    
    XYs = np.stack(
        [
            CorrelationMatrixFlat,
            WeightFlat,
        ],
        axis=1
    ) # [NeuronNumA * NeuronNumB, (Correlation, Weight)]

    Title = "Weight - ResponseSimilarity"
    utils_torch.plot.PlotPoints(
        ax, XYs, Color="Blue", Type="EmptyCircle", Size=0.5,
        XLabel="Response Similarity", YLabel="Connection Strength", 
        Title=Title,
    )

    ax = utils_torch.plot.GetAx(axes, 1)
    BinStats = utils_torch.math.CalculateBinnedMeanAndStd(CorrelationMatrixFlat, WeightFlat)
    
    utils_torch.plot.PlotMeanAndStdCurve(
        ax, BinStats.BinCenters, BinStats.Mean, BinStats.Std,
        XLabel = "Response Similarity", YLabel="Connection Strength", Title="Weight - Response Similarity Binned Mean And Std",
    )
    
    plt.suptitle(SaveName)
    plt.tight_layout()
    # Scatter plot points num might be very large, so saving in .svg might cause unsmoothness when viewing.
    utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + SaveName + "-Weight-Response-Similarity.png")
    return

def AnalyzeResponseSimilarityAndWeightUpdateCorrelation(
        ResponseA, ResponseB, WeightUpdate, Weight, 
        WeightUpdateMeasure="Ratio",
        SaveDir=None, SaveName=None, 
    ):

    # ResponseA: [BatchSize, TimeNum, NeuronNumA]
    # ResponseB: [BatchSize, TimeNum, NeuronNumB]
    ResponseA = utils_torch.ToNpArray(ResponseA)
    ResponseB = utils_torch.ToNpArray(ResponseB)
    ResponseA = ResponseA.reshape(-1, ResponseA.shape[-1])
    ResponseB = ResponseB.reshape(-1, ResponseB.shape[-1])

    Weight = utils_torch.ToNpArray(Weight)
    WeightFlat = utils_torch.FlattenNpArray(Weight)
    WeightUpdate = utils_torch.ToNpArray(WeightUpdate)
    if WeightUpdateMeasure in ["Sign"]:
        WeightUpdate = WeightUpdate / np.sign(Weight)
        WeightUpdate = utils_torch.math.ReplaceNaNOrInfWithZeroNp(WeightUpdate)
    elif WeightUpdateMeasure in ["Ratio"]:
        WeightUpdate = WeightUpdate / Weight # Ratio
        WeightUpdate = utils_torch.math.ReplaceNaNOrInfWithZeroNp(WeightUpdate)
    else:
        raise Exception(WeightUpdateMeasure)
    
    WeightUpdateFlat = utils_torch.FlattenNpArray(WeightUpdate)
    WeightUpdateStat = utils_torch.math.NpStatistics(WeightUpdate)

    CorrelationMatrix = utils_torch.math.CalculatePearsonCoefficientMatrix(ResponseA, ResponseB)
    CorrelationMatrixFlat = utils_torch.FlattenNpArray(CorrelationMatrix)

    assert CorrelationMatrix.shape == WeightUpdate.shape

    XYs = np.stack(
        [
            CorrelationMatrixFlat,
            WeightUpdateFlat
        ],
        axis=1
    ) # [NeuronNumA * NeuronNumB, (Correlation, WeightUpdate)]
    
    fig, axes = utils_torch.plot.CreateFigurePlt(4, Size="Medium")
    if WeightUpdateMeasure in ["Sign"]:
        WeightUpdateName = r'$-\frac{\partial L}{\partial w} \cdot {\rm Sign}(w) $' #r is necessary
        YRange = None
    elif WeightUpdateMeasure in ["Ratio"]:
        WeightUpdateName = r'$-\frac{\partial L}{\partial w} / w $'
        YRange = [
            WeightUpdateStat.Mean - 3.0 * WeightUpdateStat.Std,
            WeightUpdateStat.Mean + 3.0 * WeightUpdateStat.Std,
        ]
    else:
        raise Exception(WeightUpdateMeasure)
        
    ax = utils_torch.plot.GetAx(axes, 0)
    Title = "%s- ResponseSimilarity"%WeightUpdateName
    utils_torch.plot.PlotPoints(
        ax, XYs, Color="Blue", Type="EmptyCircle", Size=0.5,
        XLabel="Response Similarity", YLabel=WeightUpdateName, 
        Title=Title, YRange=YRange
    )

    ax = utils_torch.plot.GetAx(axes, 1)
    
    XYs = np.stack(
        [
            CorrelationMatrixFlat,
            WeightFlat,
        ],
        axis=1
    ) # [NeuronNumA * NeuronNumB, (Correlation, Weight)]

    Title = "Weight - ResponseSimilarity"
    utils_torch.plot.PlotPoints(
        ax, XYs, Color="Blue", Type="EmptyCircle", Size=0.5,
        XLabel="Response Similarity", YLabel="Connection Strength", 
        Title=Title,
    )

    ax = utils_torch.plot.GetAx(axes, 2)
    Title = "%s - Response Similarity Binned Mean And Std"%WeightUpdateName
    BinStats = utils_torch.math.CalculateBinnedMeanAndStd(CorrelationMatrixFlat, WeightUpdateFlat)
    utils_torch.plot.PlotMeanAndStdCurve(
        ax, BinStats.BinCenters, BinStats.Mean, BinStats.Std,
        XLabel = "Response Similarity", YLabel=WeightUpdateName, Title=Title
    )

    ax = utils_torch.plot.GetAx(axes, 3)
    BinStats = utils_torch.math.CalculateBinnedMeanAndStd(CorrelationMatrixFlat, WeightFlat)
    
    utils_torch.plot.PlotMeanAndStdCurve(
        ax, BinStats.BinCenters, BinStats.Mean, BinStats.Std,
        XLabel = "Response Similarity", YLabel="Connection Strength", Title="Weight - Response Similarity Binned Mean And Std",
    )
    
    plt.suptitle(SaveName)
    plt.tight_layout()
    # Scatter plot points num might be very large, so saving in .svg might cause unsmoothness when viewing.
    utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + SaveName + "-Weight-Response-Similarity.png")
    return