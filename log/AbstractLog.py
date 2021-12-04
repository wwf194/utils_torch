import utils_torch

class AbstractLog:
    def __init__(self, DataOnly=True):
        if DataOnly:
            self.ToFile = super().ToFileDataOnly
            self.FromFile = super().FromFileDataOnly
        else:
            self.ToFile = super().ToFileParamAndData
            self.FromFile = super().FromFileParamAndData
    def ToFileDataOnly(self, FilePath):
        if not FilePath.endswith(".data"):
            FilePath += ".data"
        utils_torch.file.PyObj2DataFile(self.data, FilePath)
        return self
    def FromFileDataOnly(self, FilePath):
        assert FilePath.endswith(".data")
        self.data = utils_torch.file.DataFile2PyObj(FilePath)
        return self
    def ToFileParamAndData(self, SaveName=None, SaveDir=None):
        if SaveName is None:
            SaveName = self.param.FullName
        utils_torch.file.PyObj2DataFile(self.data,  SaveDir + SaveName + ".data")
        utils_torch.file.PyObj2JsonFile(self.param, SaveDir + SaveName + ".jsonc")
        return self
    def FromFileParamAndData(self, SaveName, SaveDir):
        self.data  = utils_torch.file.DataFile2PyObj(SaveDir + SaveName + ".data")
        self.param = utils_torch.file.JsonFile2PyObj(SaveDir + SaveName + ".jsonc")
        self.cache = utils_torch.EmptyPyObj()
        return self

class AbstractLogAlongEpochBatchTrain(AbstractLog):
    def __init__(self, **kw):
        super().__init__(**kw)
        return

class AbstractLogForBatches(AbstractLog):
    def __init__(self, **kw):
        super().__init__(**kw)
        return
