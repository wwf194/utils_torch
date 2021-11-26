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