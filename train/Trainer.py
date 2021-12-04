import utils_torch
from utils_torch.attrs import *

class TrainerForEpochBatchTrain(utils_torch.module.AbstractModuleForEpochBatchTrain):
    def __init__(self, param, **kw):
        utils_torch.module.InitForNonModel(self, param, **kw)
    def InitFromParam(self, IsLoad=False):
        utils_torch.module.InitFromParamForModule(self, IsLoad)
        param = self.param
        cache = self.cache
        data = self.data
        
        Modules = self.Modules
        Modules.LogTrain = utils_torch.log.LogForEpochBatchTrain()
        cache.LogTrain = utils_torch.log.LogForEpochBatchTrain()

        Modules.LogTest = utils_torch.log.LogForEpochBatchTrain()
        cache.LogTest = utils_torch.log.LogForEpochBatchTrain()    

        cache.NotifyEpochBatchList = []
        cache.CheckPointList = []
        self.Register2NotifyEpochBatchList([cache.LogTrain, cache.LogTest])
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
    def ClearBatch(self):
        self.cache.BatchIndex = 0
    def ClearEpoch(self):
        self.cache.EpochIndex = 0
    def AddBatchIndex(self):
        self.cache.BatchIndex += 1
    def AddEpochIndex(self):
        self.cache.EpochIndex += 1
    def NotifyEpochIndex(self):
        cache = self.cache
        for Obj in self.cache.NotifyEpochBatchList:
            Obj.SetEpochIndex(cache.EpochIndex)
    def NotifyBatchIndex(self):
        cache = self.cache
        for Obj in self.cache.NotifyEpochBatchList:
            Obj.SetBatchIndex(cache.BatchIndex)
    def NotifyEpochNum(self):
        cache = self.cache
        for Obj in self.cache.NotifyEpochBatchList:
            Obj.SetEpochNum(cache.EpochNum)
    def NotifyBatchNum(self):
        cache = self.cache
        for Obj in self.cache.NotifyEpochBatchList:
            Obj.SetBatchNum(cache.BatchNum)
    def Register2NotifyEpochBatchList(self, List):
        cache = self.cache
        #cache.NotifyEpochBatchList = []
        for Obj in List:
            Obj = utils_torch.parse.ResolveStr(Obj)
            cache.NotifyEpochBatchList.append(Obj)
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

#utils_torch.module.SetMethodForNonModelClass(TrainerForEpochBatchTrain)
#utils_torch.module.SetEpochBatchMethodForModule(TrainerForEpochBatchTrain)