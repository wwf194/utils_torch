import utils_torch
from utils_torch.attrs import *

class AbstractModuleAlongEpochBatchTrain(utils_torch.module.AbstractModule):
    # Child Class: trainer, log
    def __init__(self, ChildClass, **kw):
        MountLocation = kw.setdefault("MountLocation", "data")
        super().__init__(**kw)
        utils_torch.train.SetEpochBatchMethodForModule(ChildClass, **kw)
    def SetEpochBatchIndexData(self, EpochIndex, BatchIndex):
        self.data.EpochIndex = EpochIndex
        self.data.BatchIndex = BatchIndex
    def SetEpochBatchIndexCache(self, EpochIndex, BatchIndex):
        self.cache.EpochIndex = EpochIndex
        self.cache.BatchIndex = BatchIndex

class TrainerEpochBatch(AbstractModuleAlongEpochBatchTrain):
    def __init__(self, param, **kw):
        super().__init__(self.__class__, MountLocation="data")
        utils_torch.transform.InitForNonModel(self, param, **kw)
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        param = self.param
        cache = self.cache
        data = self.data
        
        Modules = self.Modules
        Modules.LogTrain = utils_torch.log.LogForEpochBatchTrain()
        cache.LogTrain = utils_torch.log.LogForEpochBatchTrain()

        Modules.LogTest = utils_torch.log.LogForEpochBatchTrain()
        cache.LogTest = utils_torch.log.LogForEpochBatchTrain()    

        cache.SetEpochBatchList = []
        cache.CheckPointList = []
        self.Register2SetEpochBatchList([cache.LogTrain, cache.LogTest])
        self.BuildModules()
        self.InitModules()
        self.ParseRouters()
        self.ClearEpoch()
        self.ClearBatch()
        self.RegisterCheckPoint()
    def RegisterCheckPoint(self):
        # Scan all modules and add checkpoints among them to CheckPointList.
        cache = self.cache
        cache.CheckPointList = []
        for Name, Module in ListAttrsAndValues(cache.Modules, Exceptions=["__ResolveRef__", "__Entry__"]):
            if hasattr(Module, "IsCheckPoint") and Module.IsCheckPoint is True:
                self.cache.CheckPointList.append(Module)
                if hasattr(Module, "SetBatchNum") and hasattr(cache, "BatchNum"):
                    Module.SetBatchNum(cache.BatchNum)

    def NotifyEpochIndex(self):
        cache = self.cache
        for Obj in self.cache.SetEpochBatchList:
            Obj.SetEpochIndex(cache.EpochIndex)
    def NotifyBatchIndex(self):
        cache = self.cache
        for Obj in self.cache.SetEpochBatchList:
            Obj.SetBatchIndex(cache.BatchIndex)
    def NotifyEpochNum(self):
        cache = self.cache
        for Obj in self.cache.SetEpochBatchList:
            Obj.SetEpochNum(cache.EpochNum)
    def NotifyBatchNum(self):
        cache = self.cache
        for Obj in self.cache.SetEpochBatchList:
            Obj.SetBatchNum(cache.BatchNum)
    def Register2SetEpochBatchList(self, List):
        cache = self.cache
        #cache.SetEpochBatchList = []
        for Obj in List:
            Obj = utils_torch.parse.ResolveStr(Obj)
            cache.SetEpochBatchList.append(Obj)
    def GenerateContextInfo(self):
        cache = self.cache
        return utils_torch.PyObj({
            "Trainer": self,
            "EpochNum": cache.EpochNum,
            "BatchNum": cache.BatchNum,
            "EpochIndex": cache.EpochIndex,
            "BatchIndex": cache.BatchIndex,
        })
    # def __call__(self):
    #     utils_torch.CallGraph(self.Dynamics.Main)
    def ReportEpochBatch(self):
        cache = self.cache
        utils_torch.AddLog("Epoch%d-Batch%d"%(cache.EpochIndex, cache.BatchIndex))

#utils_torch.transform.SetMethodForNonModelClass(TrainerEpochBatch)
#utils_torch.transform.SetEpochBatchMethodForModule(TrainerEpochBatch)