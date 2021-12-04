import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *
class AbstractModule:
    def OverwriteParam(self, ParamPath, Value):
        SetAttrs(self.param, ParamPath, value=Value)

class AbstractModuleForEpochBatchTrain(AbstractModule):
    def SetEpochBatchIndexData(self, EpochIndex, BatchIndex):
        self.data.EpochIndex = EpochIndex
        self.data.BatchIndex = BatchIndex

    def SetEpochBatchIndexCache(self, EpochIndex, BatchIndex):
        self.cache.EpochIndex = EpochIndex
        self.cache.BatchIndex = BatchIndex

def SetEpochBatchMethodForModule(Class, **kw):
    MountLocation = kw.setdefault("MountLocation", "cache")
    if MountLocation in ["Cache", "cache"]:
        if not hasattr(Class, "SetEpochIndex"):
            Class.SetEpochIndex = lambda self, EpochIndex:setattr(self.cache, "EpochIndex", EpochIndex)
        if not hasattr(Class, "SetBatchIndex"):
            Class.SetBatchIndex = lambda self, EpochIndex:setattr(self.cache, "BatchIndex", EpochIndex)
        if not hasattr(Class, "SetEpochNum"):
            Class.SetEpochNum = lambda self, EpochNum:setattr(self.cache, "EpochNum", EpochNum)
        if not hasattr(Class, "SetBatchNum"):
            Class.SetBatchNum = lambda self, BatchNum:setattr(self.cache, "BatchNum", BatchNum)
        if not hasattr(Class, "GetEpochIndex"):
            Class.GetEpochIndex = lambda self:self.cache.EpochIndex
        if not hasattr(Class, "GetBatchIndex"):
            Class.GetBatchIndex = lambda self:self.cache.BatchIndex
        if not hasattr(Class, "GetEpochNum"):
            Class.GetEpochNum = lambda self:self.cache.EpochNum
        if not hasattr(Class, "GetBatchNum"):
            Class.GetBatchNum = lambda self:self.cache.BatchNum
    elif MountLocation in ["Data", "data"]:
        if not hasattr(Class, "SetEpochIndex"):
            Class.SetEpochIndex = lambda self, EpochIndex:setattr(self.data, "EpochIndex", EpochIndex)
        if not hasattr(Class, "SetBatchIndex"):
            Class.SetBatchIndex = lambda self, EpochIndex:setattr(self.data, "BatchIndex", EpochIndex)
        if not hasattr(Class, "SetEpochNum"):
            Class.SetEpochNum = lambda self, EpochNum:setattr(self.data, "EpochNum", EpochNum)
        if not hasattr(Class, "SetBatchNum"):
            Class.SetBatchNum = lambda self, BatchNum:setattr(self.data, "BatchNum", BatchNum)
        if not hasattr(Class, "GetEpochIndex"):
            Class.GetEpochIndex = lambda self:self.data.EpochIndex
        if not hasattr(Class, "GetBatchIndex"):
            Class.GetBatchIndex = lambda self:self.data.BatchIndex
        if not hasattr(Class, "GetEpochNum"):
            Class.GetEpochNum = lambda self:self.data.EpochNum
        if not hasattr(Class, "GetBatchNum"):
            Class.GetBatchNum = lambda self:self.data.BatchNum
        # if not hasattr(Class, "SetEpochBatchIndex"):
        #     Class.SetEpochBatchIndex = SetEpochBatchIndexForModuleData
    else:
        raise Exception(MountLocation)
SetEpochBatchMethodForModule(AbstractModuleForEpochBatchTrain)

class AbstractTransformModule(AbstractModule):
    def BuildModules(self):
        # initialize modules
        # for module in ListAttrs(param.modules):
        param = self.param
        cache = self.cache
        for Name, ModuleParam in ListAttrsAndValues(param.Modules, Exceptions=["__ResolveBase__"]):
            ModuleParam.Name = Name
            ModuleParam.FullName = param.FullName + "." + Name

            if not HasAttrs(ModuleParam, "Type"):
                if HasAttrs(ModuleParam, "Name"):
                    SetAttrs(ModuleParam, "Type", GetAttrs(ModuleParam.Name))
                else:
                    raise Exception()
            if ModuleParam.Type in ["Internal", "External"]:
                continue
            if cache.IsInit:
                Module = utils_torch.module.BuildModule(ModuleParam)
            else:
                Module = utils_torch.module.BuildModule(ModuleParam, LoadDir=cache.LoadDir)
            if isinstance(Module, nn.Module) and isinstance(self, nn.Module):
                self.add_module(Name, Module)
            setattr(cache.Modules, Name, Module)

class AbstractTransformModuleWithTensor(AbstractTransformModule):
    def SetTensorLocation(self, Location):
        cache = self.cache
        cache.TensorLocation = Location
        if hasattr(cache, "Tensors"):
            for ParamIndex in cache.Tensors:
                setattr(ParamIndex[0], ParamIndex[1], ParamIndex[2].to(Location).detach().requires_grad_(True))

        if hasattr(cache, "Modules"):
            for name, module in ListAttrsAndValues(cache.Modules):
                if hasattr(module, "SetTensorLocation"):
                    module.SetTensorLocation(Location)
                else:
                    if isinstance(module, nn.Module):
                        utils_torch.AddWarning("%s is an instance of nn.Module, but has not implemented SetTensorLocation method."%name)
    def GetTensorLocation(self):
        return self.cache.TensorLocation

    def SetTrainWeight(self):
        ClearTrainWeight(self)
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

AbstractModuleWithTensor = AbstractTransformModuleWithTensor