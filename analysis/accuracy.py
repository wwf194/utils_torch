import matplotlib as mpl
from matplotlib import pyplot as plt

import utils_torch
from utils_torch.attrs import *

class LogForAccuracy:
    def __init__(self):
        self.param = utils_torch.EmptyPyObj()
        self.cache = utils_torch.EmptyPyObj()


class LogForAccuracyAlongTrain:
    def __init__(self):
        self.param = utils_torch.EmptyPyObj()
        self.cache = utils_torch.EmptyPyObj()
        param = self.param
        cache = self.cache
        
        EnsureAttrs(param, "LogBatchNum", default=5)
        cache.LogBatchNum = param.LogBatchNum
        
        cache.CorrectNumList = [0 for _ in range(cache.LogBatchNum)]
        cache.TotalNumList = [0 for _ in range(cache.LogBatchNum)]
        cache.ListIndex = 0
        return
    def Update(self, CorrectNum, TotalNum):
        cache = self.cache
        cache.CorrectNumList[cache.ListIndex] = CorrectNum
        cache.TotalNumList[cache.ListIndex] = TotalNum
        cache.ListIndex = (cache.ListIndex + 1) / cache.LogBatchNum
    def GetAccuracy(self):
        cache = self.cache
        return 1.0 * sum(cache.CorrectNumList) / sum(cache.TotalNumList)


def PlotAccuracyEpochBatch(LogTrain, LogTest=None, SaveDir=None, SaveName=None, ContextObj=None):
    XsData, YsData = [], []
    
    EpochsFloatTrain = utils_torch.log.LogDict2EpochsFloat(LogTrain, BatchNum=ContextObj["BatchNum"])
    CorrectRateTrain = LogTrain["CorrectRate"]
    fig, ax = utils_torch.plot.CreateFigurePlt()
    utils_torch.plot.PlotLineChart(
        ax, Xs=EpochsFloatTrain, Ys=CorrectRateTrain,
        PlotTicks=False, Label="Train", Color="Red",
        Title="Accuracy - Epoch", XLabel="Epoch", YLabel="Accuracy",
    )
    XsData.append(EpochsFloatTrain)
    YsData.append(CorrectRateTrain)

    if LogTest is not None:
        EpochsFloatTest = utils_torch.log.LogDict2EpochsFloat(LogTest, BatchNum=ContextObj["BatchNum"])
        CorrectRateTest = LogTest["CorrectRate"]
        utils_torch.plot.PlotLineChart(
            ax, Xs=EpochsFloatTest, Ys=CorrectRateTest,
            PlotTicks=False, Label="Test", Color="Blue",
            Title="Accuracy - Epoch", XLabel="Epoch", YLabel="Accuracy",
        )
        XsData.append(EpochsFloatTest)
        YsData.append(CorrectRateTest)

    utils_torch.plot.SetXTicksFloatFromData(ax, XsData)
    utils_torch.plot.SetYTicksFloatFromData(ax, YsData)
    ax.legend()
    plt.tight_layout()

    if SaveName is None:
        SaveName = "Accuracy~Epoch"

    utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + SaveName + ".svg")
    utils_torch.files.Table2TextFileDict(LogTrain, SavePath=SaveDir + SaveName + ".txt")
    return