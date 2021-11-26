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
        utils_torch.files.Table2TextFileDict(Log, SavePath=SaveDir + "%s~Epoch.txt"%Name)
    return

def PlotLogDictStatistics(self, Name, Log, SaveDir=None):
    utils_torch.EnsureDir(SaveDir)
    Epochs = self.GetEpochsFloatFromLogDict(Log)
    fig, ax = plt.subplots()

def AnalyzeLossEpochBatch(Logs, SaveDir, ContextObj=None):
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
    utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "Loss~Epoch.svg")
    utils_torch.files.Table2TextFileDict(LossDict, SavePath=SaveDir + "Loss~Epoch.txt")
    return

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
            PlotTicks=False, Label="Test", Color="Green",
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

def PlotResponseSimilarityAndWeightCorrelation(log, SaveDir, ContextObj):
    for Name, Data in log.data.items():
        _PlotResponseSimilarityAndWeightCorrelation(
            Data.CorrelationMatrix, Data.Weight, 
            SaveDir=SaveDir + Name + "/",
            SaveName="Epoch%d-Batch%d-%s-Weight~ResponseSimilarity"%(ContextObj.EpochIndex, ContextObj.BatchIndex, Name)
        )

def _PlotResponseSimilarityAndWeightCorrelation(CorrelationMatrix, Weight, SaveDir, SaveName):
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

def AnalyzeResponseSimilarityAndWeightCorrelation(
        ResponseA, ResponseB, WeightUpdate=None, Weight=None, 
        WeightUpdateMeasure="Value",
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
    
    if WeightUpdate is not None:
        WeightUpdate = utils_torch.ToNpArray(WeightUpdate)
        if WeightUpdateMeasure in ["Value"]:
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

class LogForPCA:
    def __init__(self, EpochIndex=None, BatchIndex=None):
        self.cache = utils_torch.EmptyPyObj()
        self.data = utils_torch.EmptyPyObj()
        cache = self.cache
        data = self.data
        data.log = utils_torch.GetDefaultDict(
            lambda:utils_torch.PyObj({
                "data":[]
            })
        )
        if EpochIndex is not None:
            self.SetEpochIndex(EpochIndex)
        if BatchIndex is not None:
            self.SetBatchIndex(BatchIndex)
    def FromFile(self, FilePath):
        self.data = utils_torch.json.DataFile2PyObj(FilePath)
        return self
    def ToFile(self, FilePath):
        utils_torch.json.PyObj2DataFile(self.data, FilePath)
        return self
    def LogBatch(self, Name, data):
        data = utils_torch.ToNpArray(data)
        data = data.reshape(-1, data.shape[-1]) # [SampleNum, FeatureNum]
        self.data.log[Name.replace(".", "(dot)")].data.append(data)
    def CalculatePCA(self):
        data = self.data
        for name, log in data.log.items():
            log.data = np.concatenate(log.data, axis=0)
            log.PCATransform = utils_torch.math.PCA(log.data)
        return
utils_torch.module.SetEpochBatchMethodForModule(LogForPCA, MountLocation="data")

class LogForPCAAlongTrain:
    def __init__(self, EpochNum, BatchNum):
        #ConnectivityPattern = utils_torch.EmptyPyObj()
        data = self.data = utils_torch.EmptyPyObj()
        data.EpochNum = EpochNum
        data.BatchNum = BatchNum
        data.log = utils_torch.GetDefaultDict(lambda:[])
    def FromLogForPCA(self, logPCA, EpochIndex=None, BatchIndex=None):
        if EpochIndex is None:
            EpochIndex = logPCA.GetEpochIndex()
        if BatchIndex is None:
            BatchIndex = logPCA.GetBatchIndex()
        for Name, Log in logPCA.data.log.Items():
            self.Log(
                Name, EpochIndex, BatchIndex, Log.PCATransform
            )
    def Log(self, Name, EpochIndex, BatchIndex, PCATransform):
        _data = utils_torch.PyObj({
            "EpochIndex": EpochIndex, 
            "BatchIndex": BatchIndex,
        }).FromPyObj(PCATransform)
        self.data.log[Name].append(_data)
        return self
    def CalculateEffectiveDimNum(self, Data, RatioThres=0.5):
        # if not hasattr(Data, "VarianceExplainedRatioAccumulated"):
        #     Data.VarianceExplainedRatioAccumulated = np.cumsum(Data.VarianceExplainedRatio)
        DimCount = 0
        for RatioAccumulated in Data.VarianceExplainedRatioAccumulated:
            DimCount += 1
            if RatioAccumulated >= RatioThres:
                return DimCount
    def Plot(self, SaveDir):
        for Name, Data in self.data.log.items():
            _Name = Name.replace("(dot)", ".")
            self._Plot(
                Data,
                SaveDir + _Name + "/",
                _Name
            )
    def _Plot(self, Data, SaveDir, SaveName):
        BatchNum = self.data.BatchNum
        Data.sort(key=lambda Item:Item.EpochIndex + Item.BatchIndex * 1.0 / BatchNum)
        #SCacheSavePath = SaveDir + "Data/" + "EffectiveDimNums.data"
        # if utils_torch.files.ExistsFile(CacheSavePath):
        #     EffectiveDimNums = utils_torch.json.DataFile2PyObj(CacheSavePath)
        # else:
        for _Data in Data:
            _Data.VarianceExplainedRatioAccumulated = np.cumsum(_Data.VarianceExplainedRatio)
            _Data.FromDict({
                "EffectiveDimNums":{
                    "P100": len(_Data.VarianceExplainedRatio),
                    "P099": self.CalculateEffectiveDimNum(_Data, 0.99),
                    "P095": self.CalculateEffectiveDimNum(_Data, 0.95),
                    "P080": self.CalculateEffectiveDimNum(_Data, 0.80),
                    "P050": self.CalculateEffectiveDimNum(_Data, 0.50),
                }
            })
            utils_torch.json.PyObj2DataFile(
                _Data, SaveDir + "cache/" + "Epoch%d-Batch%d.data"%(_Data.EpochIndex, _Data.BatchIndex)
            )
        EpochIndices, BatchIndices, EpochFloats = [], [], []
        EffectiveDimNums = defaultdict(lambda:[])
        for _Data in Data:
            EpochIndices.append(_Data.EpochIndex)
            BatchIndices.append(_Data.BatchIndex)
            EpochFloats.append(_Data.EpochIndex + _Data.BatchIndex * 1.0 / BatchNum)
            EffectiveDimNums["100"].append(_Data.EffectiveDimNums.P100)
            EffectiveDimNums["099"].append(_Data.EffectiveDimNums.P099)
            EffectiveDimNums["095"].append(_Data.EffectiveDimNums.P095)
            EffectiveDimNums["080"].append(_Data.EffectiveDimNums.P080)
            EffectiveDimNums["050"].append(_Data.EffectiveDimNums.P050)

        fig, axes = utils_torch.plot.CreateFigurePlt(1)
        ax = utils_torch.plot.GetAx(axes, 0)
        LineNum = len(EffectiveDimNums)
        utils_torch.plot.SetMatplotlibParamToDefault()
        utils_torch.plot.PlotMultiLineChart(
            ax, 
            [EpochFloats for _ in range(LineNum)],
            EffectiveDimNums.values(),
            XLabel="Epochs", YLabel="DimNum required to explain proportion of total variance",
            Labels = ["$100\%$", "$99\%$", "$95\%$", "$80\%$", "$50\%$"],
            Title = "Effective Dimension Num - Training Process"
        )
        utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + SaveName + ".svg")

def AnalyzePCAForEpochBatchTrain(ContextObj):
    TestBatchNum = ContextObj.setdefault("TestBatchNum", 10)
    # Do supplementary analysis for all saved models under main save directory.
    GlobalParam = utils_torch.GetGlobalParam()
    ContextObj.setdefault("ObjRoot", GlobalParam)
    
    AnalysisSaveDir = ContextObj.setdefault("SaveDir", utils_torch.GetMainSaveDir() + "PCA-Analysis-Along-Training-Test/")

    utils_torch.DoTasks( # Dataset can be reused.
        "&^param.task.BuildDataset", **ContextObj.ToDict()
    )

    SaveDirs = utils_torch.GetAllSubSaveDirsEpochBatch(Name="SavedModel")
    
    Trainer = ContextObj.Trainer
    EpochNum = Trainer.GetEpochNum()
    
    BatchSize = Trainer.GetBatchSize()
    #BatchNum = GlobalParam.object.image.EstimateBatchNum(BatchSize, Type="Train")
    BatchNum = Trainer.GetBatchNum()

    logPCA = LogForPCAAlongTrain(EpochNum, BatchNum)
    for SaveDir in SaveDirs:
        EpochIndex, BatchIndex = utils_torch.train.ParseEpochBatchFromStr(SaveDir)
        CacheSavePath = AnalysisSaveDir + "cache/" + "Epoch%d-Batch%d.data"%(EpochIndex, BatchIndex)
        if utils_torch.ExistsFile(CacheSavePath): # Using cached data
            Data = utils_torch.json.DataFile2PyObj(CacheSavePath)
            if hasattr(Data, "PCATransform"):
                logPCA.Log(
                    EpochIndex, BatchIndex, Data.PCATransform
                )
            else:
                logPCA.Data.append(Data)
            continue
        utils_torch.AddLog("Testing Model at Epoch%d-Batch%d"%(EpochIndex, BatchIndex))

        utils_torch.DoTasks(
            "&^param.task.Load",
            In={"SaveDir": SaveDir}, 
            **ContextObj.ToDict()
        )
        utils_torch.DoTasks(
            "&^param.task.BuildTrainer", **ContextObj.ToDict()
        )
        _logPCA = RunBatchesAndCalculatePCA( # Run test batches and do PCA.
            EpochIndex=EpochIndex, BatchIndex=BatchIndex, TestBatchNum=TestBatchNum
        )
        utils_torch.json.PyObj2DataFile(
            utils_torch.PyObj({
                "PCATransform": _logPCA.PCATransform,
            }),
            CacheSavePath
        )
        logPCA.Log(
            EpochIndex, BatchIndex, _logPCA.PCATransform
        )
    logPCA.Plot(
        SaveDir=AnalysisSaveDir, SaveName="agent.model.FiringRates"
    )

def PlotPCAAlongTrain(LogsPCA, DataDir=None, SaveDir=None, ContextObj=None):
    LogPCAAlongTrain = LogForPCAAlongTrain(ContextObj.EpochNum, ContextObj.BatchNum)
    for Log in LogsPCA:
        LogPCAAlongTrain.FromLogForPCA(Log)
    LogPCAAlongTrain.Plot(SaveDir)

def ScanLogPCA(ScanDir=None):
    if ScanDir is None:
        ScanDir = utils_torch.GetMainSaveDir() + "PCA-Analysis-Along-Train-Test/" + "cache/"
    DataFiles = utils_torch.files.ListFiles(ScanDir)
    Logs = []
    for FileName in DataFiles:
        assert FileName.endswith(".data")
        Logs.append(LogForPCA().FromFile(ScanDir + FileName))
        # EpochIndex, BatchIndex = utils_torch.train.ParseEpochBatchFromStr(FileName)
        # Logs.append({
        #     "Epoch": EpochIndex,
        #     "Batch": BatchIndex,
        #     "Value": utils_torch.analysis.LogForPCA().FromFile(ScanDir + FileName)
        # })
    # Logs.sort(cmp=utils_torch.train.CmpEpochBatchDict)
    utils_torch.SortListByCmpMethod(Logs, utils_torch.train.CmpEpochBatchObj)
    return Logs

def RunBatchesAndCalculatePCA(ContextObj):
    GlobalParam = utils_torch.GetGlobalParam()
    Trainer = ContextObj.Trainer
    agent = Trainer.agent
    Dataset = Trainer.world
    BatchParam = GlobalParam.param.task.Train.BatchParam
    Dataset.PrepareBatches(BatchParam, "Test")
    log = utils_torch.log.LogForEpochBatchTrain()
    log.SetEpochIndex(0)
    TestBatchNum = ContextObj.setdefault("TestBatchNum", 10)
    logPCA = LogForPCA()
    for TestBatchIndex in range(TestBatchNum):
        utils_torch.AddLog("Epoch%d-Index%d-TestBatchIndex-%d"%(ContextObj.EpochIndex, ContextObj.BatchIndex, TestBatchIndex))
        log.SetBatchIndex(TestBatchIndex)
        InList = [
            Trainer.GetBatchParam(), Trainer.GetOptimizeParam(), log
        ]
        # InList = utils_torch.parse.ParsePyObjDynamic(
        #     utils_torch.PyObj([
        #         "&^param.task.Train.BatchParam",
        #         "&^param.task.Train.OptimizeParam",
        #         #"&^param.task.Train.NotifyEpochBatchList"
        #         log,
        #     ]),
        #     ObjRoot=GlobalParam
        # )
        utils_torch.CallGraph(agent.Dynamics.TestBatchRandom, InList=InList)
        logPCA.Log(
            log.GetLogValueByName("agent.model.FiringRates")[:, -1, :],
        )
    logPCA.ApplyPCA()
    return logPCA

from utils_torch.analysis.accuracy import LogForAccuracy, LogForAccuracyAlongTrain