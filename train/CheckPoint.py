import utils_torch
from utils_torch.attrs import *

class CheckPointForEpochBatchTrain:
    def __init__(self, param, **kw):
        utils_torch.module.InitForNonModel(self, param, ClassPath="utils_torch.Train.CheckPointForEpochBatchTrain", **kw)
    def InitFromParam(self, IsLoad):
        utils_torch.module.InitFromParamForNonModel(self, IsLoad)
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
        else:
            self.NotifyEndOfEpoch = self.NotifyEndOfEpochAlwaysFalse

        self.AddEpoch = self.AddEpochAlwaysFalse

        cache.EpochIndex = 0
        cache.BatchIndex = -1
        cache.BatchIndexTotal = -1

        EnsureAttrs(param, "Method", default="&#utils_torch.functions.NullFunction")
        cache.Method = utils_torch.parse.ResolveStr(
            param.Method,
            ObjCurrent=self.param,
            ObjRoot=utils_torch.GetGlobalParam()
        )
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
    def AddBatchStatic(self, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        cache.BatchIndexTotal += 1
        IsCheckPoint = False
        if cache.BatchIndexTotal >= self.CheckPointNext:
            IsCheckPoint = True
            cache.CheckPointNextIndex += 1
            cache.CheckPointNext = cache.CheckPointList[self.CheckPointNextIndex]
        return IsCheckPoint, self.GetMethod()
    def AddBatchOnline(self, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        cache.BatchIndexTotal += 1
        cache.IntervalIndex += 1
        if cache.IntervalIndex >= cache.IntervalCurrent:
            IsCheckPoint = True
            cache.IntervalCurrent = round(cache.IntervalCurrent * cache.IntervalIncreaseCoefficient)
            if cache.IntervalCurrent > cache.IntervalMax:
                cache.IntervalCurrent = cache.IntervalMax
            cache.IntervalIndex = 0
            return IsCheckPoint, self.GetMethod()
        else:
            IsCheckPoint = False
            return IsCheckPoint, None
    def AddBatchAlwaysTrue(self, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        cache.BatchIndexTotal += 1
        return True, self.cache.Method
    def AddBatchAlwaysFalse(self, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        cache.BatchIndexTotal += 1
        return False, self.cache.Method
    def GetMethod(self):
        return self.cache.Method
    def AddEpochAlwaysFalse(self, **kw):
        cache = self.cache
        cache.EpochIndex += 1
        cache.BatchIndex = 0
        return False, None
    def NotifyEndOfEpochAlwaysTrue(self, **kw):
        return True, self.GetMethod()
    def NotifyEndOfEpochAlwaysFalse(self, **kw):
        return False, None

CheckPointForEpochBatchTrain.IsCheckPoint = True
utils_torch.module.SetEpochBatchMethodForModule(CheckPointForEpochBatchTrain)