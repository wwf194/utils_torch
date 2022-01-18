import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# from utils_torch.module import AbstractModule, AbstractModuleWithParam, AbstractModuleWithoutParam
import utils_torch
from utils_torch.attr import *

def ProcessLogData(data):
    if isinstance(data, torch.Tensor):
        data = utils_torch.Tensor2NumpyOrFloat(data)
    return data

class AbstractTransform(utils_torch.module.AbstractModuleWithParam):
    def __init__(self, **kw):
        # kw.setdefault("DataOnly", False)
        super().__init__(**kw)
    def InitModules(self, IsLoad=False):
        cache = self.cache
        for name, module in ListAttrsAndValues(cache.Modules, Exceptions=["__ResolveBase__"]):
            if hasattr(module, "Build"):
                module.Build(IsLoad=IsLoad)
            else:
                if HasAttrs(module, "param.ClassPath"):
                    Class = module.param.ClassPath
                else:
                    Class = type(module)
                if not utils_torch.IsFunction(module):
                    utils_torch.AddWarning(
                        "Module %s of class %s has not implemented Build method."%(name, Class)
                    )
                if module is None:
                    raise Exception(name)
        for name, module in ListAttrsAndValues(cache.Dynamics, Exceptions=["__ResolveBase__"]):
            continue
    def LogCache(self, Name, data, Type=None, log=None):
        #log = utils_torch.ParseLog(log)
        data = ProcessLogData(data)
        param = self.param
        if hasattr(param, "FullName"):
            Name = param.FullName + "." + Name
        log.AddLogCache(Name, data, Type)
    def LogDict(self, Dict, Name, Type=None, log="Data"):
        log = utils_torch.ParseLog(log)
        param = self.param
        if hasattr(param, "FullName"):
            Name = param.FullName + "." + Name
        log.AddLogDict(Name, Dict, Type)
    def LogStat(self, data, Name, Type="Stat", log="Data"):
        log = utils_torch.ParseLog(log)
        param = self.param
        if hasattr(param, "FullName"):
            Name = param.FullName + "." + Name
        data = utils_torch.ToNpArray(data)
        stat = utils_torch.math.NpStatistics(data, ReturnType="Dict")
        if not Name.endswith("Stat"):
            Name += "-Stat"
        log.AddLogDict(Name, stat, Type)
    def LogActivityStat(self, Name, data, Type="Activity-Stat", log=None):
        self.LogStat(data, Name, Type=Type, log=log)
    def LogWeightStat(self, Name, WeightDict, Type="Weight-Stat", log=None):
        #log = utils_torch.ParseLog(log)
        param = self.param
        for Name, Weight in WeightDict.items():
            WeightStat = utils_torch.math.TorchTensorStat(Weight, ReturnType="Dict")
            log.AddLogDict(Name + "-Stat", WeightStat, Type)
    def LogActivityAlongTime(self, Name, data, Type="ActivityAlongTime", log="Data"):
        log = utils_torch.ParseLog(log)
        param = self.param
        data = utils_torch.ToNpArray(data)
        if hasattr(param, "FullName"):
            Name = param.FullName + "." + Name
        log.AddLogCache(Name, data, Type)
    def LogTensor(self, Name, data, Type="None", log="Data"):
        log = utils_torch.ParseLog(log)
        param = self.param
        data = utils_torch.ToNpArray(data)
        if hasattr(param, "FullName"):
            Name = param.FullName + "." + Name
        log.AddLogCache(Name, data, Type)
    def LogActivityStat(self, log: utils_torch.log.LogAlongEpochBatchTrain):
        for Name, Activity in log.GetLogValueOfType("ActivityAlongTime").items():
            self.LogActivityStat(Name + "-Stat", Activity, "Activity-Stat", log)
    def LogActivity(self, Name, data, Type="Activity", log=None):
        #log = utils_torch.ParseLog(log)
        param = self.param
        data = utils_torch.ToNpArray(data)
        if hasattr(param, "FullName"):
            Name = param.FullName + "." + Name
        log.AddLogCache(Name, data, Type)
    def Log(self, Name, data, Type=None, log="Data"):
        log = utils_torch.ParseLog(log)
        param = self.param
        if hasattr(param, "FullName"):
            Name = param.FullName + "." + Name
        data = ProcessLogData(data)
        log.AddLog(Name, data, Type)
    def LogWeight(self, Name="Weight", WeightDict=None, Type="Weight", log=None):
        self.LogTensorDict(Name, WeightDict, Type, log)
    def LogGrad(self, Name="Grad", WeightDict=None, Type="Grad", log=None):
        self.LogTensorDict(Name, WeightDict, Type, log)  
    def LogTensorDict(self, Name="None", TensorDict=None, Type="None", log=None):
        #log = utils_torch.ParseLog(log)
        param = self.param
        _weights = {}
        for name, weight in TensorDict.items():
            _weights[name] = utils_torch.ToNpArray(weight)
        log.AddLogCache(param.FullName + "." + Name, _weights, Type)
    def LogFloat(self, Name, data, Type="Float", log="Data"):
        log = utils_torch.ParseLog(log)
        param = self.param
        if isinstance(data, torch.Tensor):
            data = data.item()
        if hasattr(param, "FullName"):
            Name = param.FullName + "." + Name
        log.AddLog(Name, data, Type)
    def LogLoss(self, Name, loss, Type="Loss", log="Data"):
        log = utils_torch.ParseLog(log)
        if isinstance(loss, torch.Tensor):
            data = loss.item()
        log.AddLog(Name, data, Type)
    def LogLossDict(self, Name, LossDict, Type="Loss", log=None):
        # #log = utils_torch.ParseLog(log)
        # for Name, Loss in LossDict.items():
        #     if isinstance(Loss, torch.Tensor):
        #         LossDict[Name] = Loss.item()
        # log.AddLogDict(Name, LossDict, Type)
        for Name, Loss in LossDict.items():
            self.LogLoss(Name, Loss, Type, log)
class AbstractTransformWithTensor(AbstractTransform):
    def __init__(self, **kw):
        super().__init__(**kw)

    def BeforeBuild(self, IsLoad=False):
        super().BeforeBuild(IsLoad=IsLoad)
        cache = self.cache
        cache.Tensors = utils_torch.PyObj([])

    def ClearTrainWeight(self):
        utils_torch.RemoveAttrIfExists(self.cache, "TrainWeight")
    def SetTrainWeight(self):
        self.ClearTrainWeight()
        cache = self.cache
        cache.TrainWeight = {}
        if hasattr(cache, "Modules"):
            for ModuleName, Module in utils_torch.ListAttrsAndValues(cache.Modules):
                if hasattr(Module, "SetTrainWeight"):
                    TrainWeight = Module.SetTrainWeight()
                    for name, weight in TrainWeight.items():
                        cache.TrainWeight[ModuleName + "." + name] = weight
                else:
                    if isinstance(Module, nn.Module):
                        utils_torch.AddWarning("Module %s is instance of nn.Module, but has not implemented GetTrainWeight method."%Module)
            return cache.TrainWeight
        else:
            return {}
    def GetTrainWeight(self):
        return self.cache.TrainWeight
    def GetPlotWeight(self):
        cache = self.cache
        if not hasattr(cache, "PlotWeight"):
            self.SetPlotWeight()
        weights = {}
        for name, method in cache.PlotWeight.items():
            weights[name] = method()
        return weights

    def SetPlotWeight(self):
        self.ClearPlotWeight()
        cache = self.cache
        cache.PlotWeight = {}
        if hasattr(cache, "Modules"):
            for ModuleName, Module in utils_torch.ListAttrsAndValues(cache.Modules):
                if hasattr(Module, "GetPlotWeight"):
                    PlotWeightMethod = Module.SetPlotWeight()
                    for name, method in PlotWeightMethod.items():
                        cache.PlotWeight[ModuleName + "." + name] = method
                else:
                    if isinstance(Module, nn.Module):
                        utils_torch.AddWarning("Module %s is instance of nn.Module, but has not implemented GetTrainWeight method."%Module)
            return cache.PlotWeight
        else:
            return {}

    def ClearPlotWeight(self):
        cache = self.cache
        if hasattr(cache, "PlotWeight"):
            delattr(cache, "PlotWeight")