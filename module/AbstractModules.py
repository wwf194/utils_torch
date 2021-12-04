import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *
class AbstractModule:
    def OverwriteParam(self, ParamPath, Value):
        SetAttrs(self.param, ParamPath, value=Value)

class AbstractModuleWithTensor(AbstractModule):
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