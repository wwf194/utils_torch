import torch
import numpy as np
import scipy
import matplotlib as mpl
from matplotlib import pyplot as plt
from collections import defaultdict

import utils_torch
from utils_torch.attrs import *

def AnalyzeTimeVaryingActivitiesEpochBatch(Logs, PlotIndex=0, SaveDir=None, ContextObj=None):
    PlotIndex = 0
    for name, activity in Logs.items():
        EpochIndex = activity["Epoch"]
        BatchIndex = activity["Batch"]
        activity = activity["Value"]
        _name = "%s-Epoch%d-Batch%d-No%d"%(name, ContextObj.EpochIndex, ContextObj.BatchIndex, PlotIndex)
        utils_torch.analysis.AnalyzeActivityAlongTime(
            activity,
            PlotIndex=PlotIndex,
            Name=_name, 
            SavePath=SaveDir + "%s/%s.svg"%(name, _name),
        )

def AnalyzeActivityAlongTime(activity, PlotIndex, Name=None, SavePath=None):
    utils_torch.plot.PlotActivityAndDistributionAlongTime(
        axes=None,
        activity=activity,
        activityPlot=activity[PlotIndex],
        Title=Name,
        Save=True,
        SavePath=SavePath,
    )

def AnalyzeWeightsEpochBatch(Logs, BatchIndex=None, PlotIndex=0, SaveDir=None, ContextObj=None):
    PlotIndex = 0
    weights = Logs["Value"]
    EpochIndex = Logs["Epoch"]
    BatchIndex = Logs["Batch"]
    for name, weight in weights.items():
        _name = "Epoch%d-Batch%d-No%d-%s"%(EpochIndex, BatchIndex, PlotIndex, name)
        utils_torch.analysis.AnalyzeWeight(
            weight,
            Name=_name, 
            SavePath=SaveDir + name + "/" + "%s.svg"%(_name),
        )

def AnalyzeWeight(weight, Name, SavePath=None):
    utils_torch.plot.PlotWeightAndDistribution(
        axes=None,
        weight=weight,
        Name=Name,
        SavePath=SavePath,
    )
    return

def AnalyzeStatAlongTrainEpochBatch(Logs, SaveDir, ContextObj):
    for Name, Log in Logs.items(): # Each Log is statistics of a weight along training process.
        assert isinstance(Log, dict)
        EpochIndices = Log["Epoch"]
        BatchIndices = Log["Batch"]
        EpochsFloat = utils_torch.train.EpochBatchIndices2EpochsFloat(
            EpochIndices, BatchIndices, BatchNum = ContextObj["BatchNum"]
        )
        fig, ax = utils_torch.plot.CreateFigurePlt()
        utils_torch.plot.PlotMeanAndStdCurve(
            ax, Xs=EpochsFloat,
            Mean=Log["Mean"], Std=Log["Std"], 
            Title="%s - Epoch"%Name, XLabel="Epoch", YLabel=Name,
        )
        plt.tight_layout()
        utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "%s~Epoch.svg"%Name)
        utils_torch.file.Table2TextFileDict(Log, SavePath=SaveDir + "%s~Epoch.txt"%Name)
    return

def PlotLogDictStatistics(self, Name, Log, SaveDir=None):
    utils_torch.EnsureDir(SaveDir)
    Epochs = self.GetEpochsFloatFromLogDict(Log)
    fig, ax = plt.subplots()

def PlotAllLossEpochBatch(Logs, SaveDir, SaveName=None, ContextObj=None):
    assert len(Logs) > 0
    EpochsFloat = utils_torch.log.LogList2EpochsFloat(list(Logs.values())[0], BatchNum=ContextObj["BatchNum"])
    LossDict = {}
    for Name, Log in Logs.items(): # Each Log is statistics of a weight along training process.
        #assert isinstance(Log, list)
        Loss = Log["Value"]
        LossDict[Name] = Loss
    fig, ax = utils_torch.plot.CreateFigurePlt()
    utils_torch.plot.PlotMultiLineChartWithSameXs(
        ax, Xs=EpochsFloat, YsDict=LossDict,
        XTicks="Float", YTicks="Float",
        Title="Loss - Epoch", XLabel="Epoch", YLabel="Loss",
    )
    plt.tight_layout()

    if SaveName is None:
        SaveName = "Loss~Epoch"
    
    utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + SaveName + ".svg")
    utils_torch.file.Table2TextFileDict(LossDict, SavePath=SaveDir + SaveName + ".txt")
    return

def PlotTotalLossEpochBatch(LogTrain, LogTest=None, SaveDir=None, SaveName=None, ContextObj=None):
    if SaveName is None:
        SaveName = "TotalLoss~Epoch"

    XsData, YsData = [], []
    EpochsFloat = utils_torch.log.LogList2EpochsFloat(LogTrain, BatchNum=ContextObj["BatchNum"])
    Loss = LogTrain["Value"]
    fig, ax = utils_torch.plot.CreateFigurePlt()
    utils_torch.plot.PlotLineChart(
        ax, Xs=EpochsFloat, Ys=Loss,
        PlotTicks=False, Label="Train", Color="Red",
        Title=SaveName, XLabel="Epoch", YLabel="Total Loss",
    )
    XsData.append(EpochsFloat)
    YsData.append(Loss)
    utils_torch.file.Table2TextFileDict(LogTrain, SavePath=SaveDir + SaveName + "(Train).txt")

    if LogTest is not None:
        EpochsFloat = utils_torch.log.LogList2EpochsFloat(LogTest, BatchNum=ContextObj["BatchNum"])
        Loss = LogTest["Value"]
        utils_torch.plot.PlotLineChart(
            ax, Xs=EpochsFloat, Ys=Loss,
            PlotTicks=False, Label="Test", Color="Blue",
            Title=SaveName, XLabel="Epoch", YLabel="Total Loss",
        )
        XsData.append(EpochsFloat)
        YsData.append(Loss)
        utils_torch.file.Table2TextFileDict(LogTest, SavePath=SaveDir + SaveName + "(Test).txt")

    utils_torch.plot.SetXTicksFloatFromData(ax, XsData)
    utils_torch.plot.SetYTicksFloatFromData(ax, YsData)
    ax.legend()
    plt.tight_layout()

    utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + SaveName + ".svg")
    return
PlotLossEpochBatch = PlotTotalLossEpochBatch

from utils_torch.analysis.accuracy import LogForAccuracy, LogForAccuracyAlongTrain, PlotAccuracyEpochBatch
from utils_torch.analysis.PCA import LogForPCA, LogForPCAAlongTrain, ScanLogPCA, AnalyzePCAForEpochBatchTrain, PlotPCAAlongTrain