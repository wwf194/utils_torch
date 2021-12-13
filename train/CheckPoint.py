import utils_torch
from utils_torch.attr import *

from utils_torch.module.AbstractModules import AbstractModule
class CheckPointForEpochBatchTrain(utils_torch.log.AbstractLogAlongEpochBatchTrain):
    def __init__(self, **kw):
        #utils_torch.transform.InitForNonModel(self, param, ClassPath="utils_torch.Train.CheckPointForEpochBatchTrain", **kw)
        super().__init__(**kw)
    def SetMethod(self, Method):
        assert callable(Method)
        self.cache.Method = Method
        self.param.Method = str(Method)
        return self
    def Build(self, IsLoad=False):
        #utils_torch.transform.BuildForNonModel(self, IsLoad)
        super().BeforeBuild(IsLoad=IsLoad)
        # Intervals are calculated in batches, not epochs.
        param = self.param
        cache = self.cache
        EnsureAttrs(param, "CalculateCheckPointMode", default="EndOfEpoch")
        if param.CalculateCheckPointMode in ["Static"]: # For cases where EpochNum and BatchNum is known before training.
            assert HasAttrs(param, "Epoch.Num")
            assert HasAttrs(param, "Batch.Num")
            EnsureAttrs(param, "Interval.IncreaseCoefficient", value=1.5)
            EnsureAttrs(param, "Interval.Start", value=10)
            EnsureAttrs(param, "Interval.Max", value=10 * param.Batch.Num)
            cache.CheckPointList = self.CalculateCheckPointList(param)
            cache.CheckPointNextIndex = 0
            cache.CheckPointNext = self.CheckPointList[self.CheckPointNextIndex]
            self.AddBatch = self.AddBatchStatic
        elif param.CalculateCheckPointMode in ["Online"]:
            # No need to know BatchNum in advance
            EnsureAttrs(param, "Interval.IncreaseCoefficient", value=1.5)
            EnsureAttrs(param, "Interval.Start", value=10)
            EnsureAttrs(param, "Interval.Max", value=10000)
            cache.IntervalCurrent = param.Interval.Start
            cache.IntervalIndex = 0
            cache.IntervalMax = param.Interval.Max
            cache.IntervalIncreaseCoefficient = param.Interval.IncreaseCoefficient
            self.AddBatch = self.AddBatchOnline
        elif param.CalculateCheckPointMode in ["Always", "EveryBatch"]:
            cache.IntervalIndex = 0
            self.AddBatch = self.AddBatchAlwaysTrue
        elif param.CalculateCheckPointMode in ["EndOfEpoch"]:
            pass
        else:
            raise Exception(param.CalculateCheckPointMode)
        
        if param.CalculateCheckPointMode in ["EndOfEpoch"]:
            self.NotifyEndOfEpoch = self.NotifyEndOfEpochAlwaysTrue
            self.AddBatch = self.AddBatchAlwaysFalse
            cache.IsCheckPoint = False
        else:
            self.NotifyEndOfEpoch = self.NotifyEndOfEpochAlwaysFalse

        self.AddEpoch = self.AddEpochAlwaysFalse

        cache.EpochIndex = 0
        cache.BatchIndex = -1
        cache.BatchIndexTotal = -1

        if hasattr(param, "Method"):
            #EnsureAttrs(param, "Method", default="&#utils_torch.functions.NullFunction")
            cache.Method = utils_torch.parse.ResolveStr(
                param.Method,
                ObjCurrent=self.param,
                ObjRoot=utils_torch.GetGlobalParam()
            )
        
        #self.SetBatchIndex = self.AddBatch
        return self
    def SetBatchIndex(self, BatchIndex):
        self.AddBatch()
    def CalculateCheckPointList(param):
        BatchNumTotal = param.Epoch.Num * param.Batch.Num
        CheckPointBatchIndices = []
        BatchIndexTotal = 0
        CheckPointBatchIndices.append(BatchIndexTotal)
        Interval = param.Interval.Start
        while BatchIndexTotal < BatchNumTotal:
            BatchIndexTotal += round(Interval)
            CheckPointBatchIndices.append(BatchIndexTotal)
            Interval *= param.Interval.IncreaseCoefficient
            if Interval > param.Interval.Max:
                Interval =param.Interval.Max
        return CheckPointBatchIndices
    def IsCheckPoint(self):
        return self.cache.IsCheckPoint
    def AddBatchStatic(self, *arg, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        cache.BatchIndexTotal += 1
        cache.IsCheckPoint = False
        if cache.BatchIndexTotal >= self.CheckPointNext:
            cache.IsCheckPoint = True
            cache.CheckPointNextIndex += 1
            cache.CheckPointNext = cache.CheckPointList[self.CheckPointNextIndex]
            return cache.IsCheckPoint, self.GetMethod()
        else:
            return cache.IsCheckPoint, None
    def AddBatchOnline(self, *arg, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        cache.BatchIndexTotal += 1
        cache.IntervalIndex += 1
        if cache.IntervalIndex >= cache.IntervalCurrent:
            cache.IsCheckPoint = True
            cache.IntervalCurrent = round(cache.IntervalCurrent * cache.IntervalIncreaseCoefficient)
            if cache.IntervalCurrent > cache.IntervalMax:
                cache.IntervalCurrent = cache.IntervalMax
            cache.IntervalIndex = 0
            return cache.IsCheckPoint, self.GetMethod()
        else:
            cache.IsCheckPoint = False
            return cache.IsCheckPoint, None
    def AddBatchAlwaysTrue(self, *arg, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        cache.BatchIndexTotal += 1
        cache.IsCheckPoint = True
        return cache.IsCheckPoint, self.cache.Method
    def AddBatchAlwaysFalse(self, *arg, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        cache.BatchIndexTotal += 1
        cache.IsCheckPoint = False
        return cache.IsCheckPoint, self.cache.Method
    def GetMethod(self):
        return self.cache.Method
    def AddEpochAlwaysFalse(self, *arg, **kw):
        cache = self.cache
        cache.EpochIndex += 1
        cache.BatchIndex = 0
        return False, None
    def NotifyEndOfEpochAlwaysTrue(self, **kw):
        return True, self.GetMethod()
    def NotifyEndOfEpochAlwaysFalse(self, **kw):
        return False, None

#CheckPointForEpochBatchTrain.IsCheckPoint = True
#utils_torch.transform.SetEpochBatchMethodForModule(CheckPointForEpochBatchTrain)