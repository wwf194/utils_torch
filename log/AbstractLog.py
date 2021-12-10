import utils_torch

class AbstractLog(utils_torch.module.AbstractModuleWithoutParam):
    def __init__(self, **kw):
        #kw.setdefault("DataOnly", True) # Log class do not need param
        super().__init__(**kw)


class AbstractLogAlongEpochBatchTrain(AbstractLog):
    def __init__(self, **kw):
        super().__init__(**kw)
        return
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
    def SetEpochNum(self, EpochNum):
        self.data.EpochNum = EpochNum
    def SetBatchNum(self, BatchNum):
        self.data.BatchNum = BatchNum
    def SetEpochIndex(self, EpochIndex):
        self.data.EpochIndex = EpochIndex
    def SetBatchIndex(self, BatchIndex):
        self.data.BatchIndex = BatchIndex
    def GetEpochNum(self):
        return self.data.EpochNum
    def GetBatchNum(self):
        return self.data.BatchNum
    def GetEpochIndex(self):
        return self.data.EpochIndex
    def GetBatchIndex(self):
        return self.data.BatchIndex
AbstractModuleAlongEpochBatchTrain = AbstractLogAlongEpochBatchTrain

class AbstractLogAlongBatch(AbstractLog):
    def __init__(self, **kw):
        super().__init__(**kw)
        return
