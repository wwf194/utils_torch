import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# from utils_torch.module import AbstractModule, AbstractModuleWithParam, AbstractModuleWithoutParam
import utils_torch
from utils_torch.attrs import *

def ProcessLogData(data):
    if isinstance(data, torch.Tensor):
        data = utils_torch.Tensor2NumpyOrFloat(data)
    return data

class AbstractTransform(utils_torch.module.AbstractModuleWithParam):
    def __init__(self, **kw):
        kw.setdefault("DataOnly", False)
        super().__init__(**kw)
    def InitModules(self):
        cache = self.cache
        for name, module in ListAttrsAndValues(cache.Modules):
            if hasattr(module, "Build"):
                module.Build(IsLoad=cache.IsLoad)
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

    def LogCache(self, data, Name, Type=None, log=None):
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
    def LogActivityStat(self, data, Name, Type="Activity-Stat", log="Data"):
        self.LogStat(self, data, Name, Type=Type, log=log)
    def LogWeightStat(self, weights, Name, Type="Weight-Stat", log="Data"):
        log = utils_torch.ParseLog(log)
        param = self.param
        for Name, Weight in weights.items():
            WeightStat = utils_torch.math.TorchTensorStat(Weight, ReturnType="Dict")
            log.AddLogDict(Name, WeightStat, Type)
    def LogActivityAlongTime(self, data, Name, Type="ActivityAlongTime", log="Data"):
        log = utils_torch.ParseLog(log)
        param = self.param
        data = utils_torch.ToNpArray(data)
        if hasattr(param, "FullName"):
            Name = param.FullName + "." + Name
        log.AddLogCache(Name, data, Type)
    def LogActivity(self, data, Name, Type="Activity", log="Data"):
        log = utils_torch.ParseLog(log)
        param = self.param
        data = utils_torch.ToNpArray(data)
        if hasattr(param, "FullName"):
            Name = param.FullName + "." + Name
        log.AddLogCache(Name, data, Type)
    def Log(self, data, Name, Type=None, log="Data"):
        log = utils_torch.ParseLog(log)
        param = self.param
        if hasattr(param, "FullName"):
            Name = param.FullName + "." + Name
        data = ProcessLogData(data)
        log.AddLog(Name, data, Type)
    def LogWeight(self, weights, Name="Weight", Type="Weight", log="Data"):
        log = utils_torch.ParseLog(log)
        param = self.param
        _weights = {}
        for name, weight in weights.items():
            _weights[name] = utils_torch.ToNpArray(weight)
        log.AddLogCache(Name, _weights, Type)
    def LogFloat(self, data, Name, Type="Float", log="Data"):
        log = utils_torch.ParseLog(log)
        param = self.param
        if isinstance(data, torch.Tensor):
            data = data.item()
        if hasattr(param, "FullName"):
            Name = param.FullName + "." + Name
        log.AddLog(Name, data, Type)
    def LogLoss(self, loss, Name, Type="Loss", log="Data"):
        log = utils_torch.ParseLog(log)
        if isinstance(loss, torch.Tensor):
            data = loss.item()
        log.AddLog(Name, data, Type)

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