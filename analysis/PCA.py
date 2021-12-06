import numpy as np
import collections

import utils_torch

class LogForPCA(utils_torch.log.AbstractLogAlongBatch):
    def __init__(self, EpochIndex=None, BatchIndex=None, **kw):
        super.__init__(**kw)
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
            self.data.EpochIndex = EpochIndex
        if BatchIndex is not None:
            self.data.BatchIndex = BatchIndex
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
#utils_torch.transform.SetEpochBatchMethodForModule(LogForPCA, MountLocation="data")

class LogForPCAAlongTrain(utils_torch.log.AbstractLogAlongEpochBatchTrain):
    def __init__(self, EpochNum, BatchNum, **kw):
        super.__init__(**kw)
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
        # if utils_torch.file.ExistsFile(CacheSavePath):
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
        EffectiveDimNums = utils_torch.GetDefaultDict(lambda:[])
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
    DataFiles = utils_torch.file.ListFiles(ScanDir)
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

# def RunBatchesAndCalculatePCA(ContextObj):
#     GlobalParam = utils_torch.GetGlobalParam()
#     Trainer = ContextObj.Trainer
#     agent = Trainer.agent
#     Dataset = Trainer.world
#     BatchParam = GlobalParam.param.task.Train.BatchParam
#     Dataset.CreateFlow(BatchParam, "Test")
#     log = utils_torch.log.LogForEpochBatchTrain()
#     log.SetEpochIndex(0)
#     TestBatchNum = ContextObj.setdefault("TestBatchNum", 10)
#     logPCA = LogForPCA()
#     for TestBatchIndex in range(TestBatchNum):
#         utils_torch.AddLog("Epoch%d-Index%d-TestBatchIndex-%d"%(ContextObj.EpochIndex, ContextObj.BatchIndex, TestBatchIndex))
#         log.SetBatchIndex(TestBatchIndex)
#         InList = [
#             Trainer.GetBatchParam(), Trainer.GetOptimizeParam(), log
#         ]
#         # InList = utils_torch.parse.ParsePyObjDynamic(
#         #     utils_torch.PyObj([
#         #         "&^param.task.Train.BatchParam",
#         #         "&^param.task.Train.OptimizeParam",
#         #         #"&^param.task.Train.NotifyEpochBatchList"
#         #         log,
#         #     ]),
#         #     ObjRoot=GlobalParam
#         # )
#         utils_torch.CallGraph(agent.Dynamics.RunTestBatch, InList=InList)
#         logPCA.Log(
#             log.GetLogValueByName("agent.model.FiringRates")[:, -1, :],
#         )
#     logPCA.ApplyPCA()
#     return logPCA
