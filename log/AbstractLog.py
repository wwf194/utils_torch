import utils_torch

class AbstractLog(utils_torch.module.AbstractModuleWithParam):
    def __init__(self, **kw):
        #kw.setdefault("DataOnly", True) # Log class do not need param
        super().__init__(**kw)
    def SetEpochNum(self, EpochNum):
        self.data.EpochNum = EpochNum
        return self
    def SetBatchNum(self, BatchNum):
        self.data.BatchNum = BatchNum
        return self
    def SetEpochIndex(self, EpochIndex):
        self.data.EpochIndex = EpochIndex
        return self
    def SetBatchIndex(self, BatchIndex):
        self.data.BatchIndex = BatchIndex
        return self
    def GetEpochNum(self):
        return self.data.EpochNum
    def GetBatchNum(self):
        return self.data.BatchNum
    def GetEpochIndex(self):
        return self.data.EpochIndex
    def GetBatchIndex(self):
        return self.data.BatchIndex
    def SetEpochBatchIndex(self, EpochIndex, BatchIndex):
        self.data.EpochIndex = EpochIndex
        self.data.BatchIndex = BatchIndex
        return self

class AbstractLogAlongEpochBatchTrain(AbstractLog):
    def __init__(self, **kw):
        super().__init__(**kw)
        return
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)

AbstractModuleAlongEpochBatchTrain = AbstractLogAlongEpochBatchTrain

class AbstractLogAlongBatch(AbstractLog):
    def __init__(self, **kw):
        super().__init__(**kw)
        return
    def Build(self, IsLoad=False):
        super().BeforeBuild(IsLoad=IsLoad)
        return self
