import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import time
import os
import re

import utils_torch
from utils_torch.attrs import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

def NotifyEpochIndex(ObjList, EpochIndex):
    for Obj in ObjList:
        Obj.NotifyEpochIndex(EpochIndex)

def NotifyBatchIndex(ObjList, BatchIndex):
    for Obj in ObjList:
        Obj.NotifyBatchIndex(BatchIndex)

def NotifyEpochNum(ObjList, EpochNum):
    for Obj in ObjList:
        Obj.NotifyEpochNum(EpochNum)

def NotifyBatchNum(ObjList, BatchNum):
    for Obj in ObjList:
        Obj.NotifyBatchNum(BatchNum)

def ParseRoutersFromTrainParam(param, **kw):
    Routers = utils_torch.PyObj()
    for Name, RouterParam in ListAttrsAndValues(param.Batch.Routers):
        Router = utils_torch.router.ParseRouterStaticAndDynamic(RouterParam, ObjRefList=[RouterParam, param], **kw)
        setattr(Routers, Name, Router)
    return Routers

def SetSaveDirForSavedModel(EpochIndex, BatchIndex):
    SaveDirForSavedModel = utils_torch.GetMainSaveDir() + "SavedModel/" + "Epoch%d-Batch%d/"%(EpochIndex, BatchIndex)
    utils_torch.SetSubSaveDir(SaveDirForSavedModel, Type="Obj")

# def CallGraphEpochBatch(router, InList, logger, EpochIndex, BatchIndex):
#     logger.SetLocal("EpochIndex", EpochIndex)
#     logger.SetLocal("BatchIndex", BatchIndex)
#     utils_torch.AddLog("Epoch%d-Batch%d"%(EpochIndex, BatchIndex))
#     utils_torch.CallGraph(router, InList=InList) 

def ParseEpochBatchFromStr(Str):
    MatchResult = re.match(r"^.*Epoch(-?\d*)-Batch(\d*).*$", Str)
    if MatchResult is None:
        raise Exception(Str)
    EpochIndex = int(MatchResult.group(1))
    BatchIndex = int(MatchResult.group(2))
    return EpochIndex, BatchIndex


def ClearGrad(weights):
    for name, weight in weights.items():
        if weight.grad is not None:
                weight.grad.detach_()
                weight.grad.zero_()

def GetEpochFloat(EpochIndex, BatchIndex, BatchNum):
    return EpochIndex + BatchIndex / BatchNum * 1.0

def EpochBatchIndices2EpochsFloat(EpochIndices, BatchIndices, **kw):
    BatchNum = kw["BatchNum"]
    EpochIndices = utils_torch.ToNpArray(EpochIndices)
    BatchIndices = utils_torch.ToNpArray(BatchIndices)
    EpochsFloat = EpochIndices + BatchIndices / BatchNum
    return utils_torch.NpArray2List(EpochsFloat)

def Labels2OneHotVectors(Labels, VectorSize=None):
    # Labels: [SampleNum]
    SampleNum = Labels.shape[0]
    Labels = utils_torch.ToNpArray(Labels, dtype=np.int32)
    if VectorSize is None:
        LabelMin, LabelMax = np.min(Labels), np.max(Labels)
        VectorSize = LabelMax
    OneHotVectors = np.zeros((SampleNum, VectorSize), dtype=np.float32)
    OneHotVectors[range(SampleNum), Labels] = 1
    return OneHotVectors


# def Probability2MaxIndex(Probability):
#     # Probability: [BatchSize, ClassNum]
#     return torch.argmax(Probability, axis=1)

def CmpEpochBatch(EpochIndex1, BatchIndex1, EpochIndex2, BatchIndex2):
    if EpochIndex1 < EpochIndex2:
        return -1
    elif EpochIndex1 > EpochIndex2:
        return 1
    else:
        if BatchIndex1 < BatchIndex2:
            return -1
        elif BatchIndex1 > BatchIndex2:
            return 1
        else:
            return 0   

def CmpEpochBatchData(data1, data2):
    EpochIndex1 = data1.EpochIndex
    BatchIndex1 = data1.BatchIndex
    EpochIndex2 = data2.EpochIndex
    BatchIndex2 = data2.BatchIndex
    return CmpEpochBatch(EpochIndex1, BatchIndex1, EpochIndex2, BatchIndex2)

def CmpEpochBatchDict(Dict1, Dict2):
    EpochIndex1 = Dict1["Epoch"]
    BatchIndex1 = Dict1["Batch"]
    EpochIndex2 = Dict2["Epoch"]
    BatchIndex2 = Dict2["Batch"]
    return CmpEpochBatch(EpochIndex1, BatchIndex1, EpochIndex2, BatchIndex2)

def CmpEpochBatchObj(Obj1, Obj2):
    EpochIndex1 = Obj1.GetEpochIndex()
    BatchIndex1 = Obj1.GetBatchIndex()
    EpochIndex2 = Obj2.GetEpochIndex()
    BatchIndex2 = Obj2.GetBatchIndex()
    return CmpEpochBatch(EpochIndex1, BatchIndex1, EpochIndex2, BatchIndex2)

def GetEpochBatchIndexFromPyObj(Obj):
    if hasattr(Obj, "Epoch"):
        EpochIndex = Obj.Epoch
    elif hasattr(Obj, "EpochIndex"):
        EpochIndex = Obj.EpochIndex
    else:
        raise Exception()
    
    if hasattr(Obj, "Batch"):
        BatchIndex = Obj.Batch
    elif hasattr(Obj, "BatchIndex"):
        BatchIndex = Obj.BatchIndex
    else:
        raise Exception()
    return EpochIndex, BatchIndex

from utils_torch.train.CheckPoint import CheckPointForEpochBatchTrain
from utils_torch.train.Trainer import TrainerEpochBatch


def ClearBatch(self, Obj):
    Obj.BatchIndex = 0
def ClearEpoch(self, Obj):
    Obj.EpochIndex = 0
def AddBatchIndex(self, Obj):
    Obj.BatchIndex += 1
def AddEpochIndex(self, Obj):
    Obj.EpochIndex += 1
    
def SetEpochBatchMethodForModule(Class, **kw):
    MountLocation = kw.setdefault("MountLocation", "cache")
    if MountLocation in ["Cache", "cache"]:
        if not hasattr(Class, "SetEpochIndex"):
            Class.SetEpochIndex = lambda self, EpochIndex:setattr(self.cache, "EpochIndex", EpochIndex)
        if not hasattr(Class, "SetBatchIndex"):
            Class.SetBatchIndex = lambda self, BatchIndex:setattr(self.cache, "BatchIndex", BatchIndex)
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
        if not hasattr(Class, "ClearBatch"):
            Class.ClearBatch = lambda self:ClearBatch(self, self.cache)
        if not hasattr(Class, "ClearEpoch"):
            Class.ClearBatch = lambda self:ClearEpoch(self, self.cache)
        if not hasattr(Class, "AddBatchIndex"):
            Class.ClearBatch = lambda self:AddBatchIndex(self, self.cache)
        if not hasattr(Class, "AddEpochIndex"):
            Class.ClearBatch = lambda self:AddEpochIndex(self, self.cache)

    elif MountLocation in ["Data", "data"]:
        if not hasattr(Class, "SetEpochIndex"):
            Class.SetEpochIndex = lambda self, EpochIndex:setattr(self.data, "EpochIndex", EpochIndex)
        if not hasattr(Class, "SetBatchIndex"):
            Class.SetBatchIndex = lambda self, BatchIndex:setattr(self.data, "BatchIndex", BatchIndex)
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
        if not hasattr(Class, "ClearBatch"):
            Class.ClearBatch = lambda self:ClearBatch(self, self.data)
        if not hasattr(Class, "ClearEpoch"):
            Class.ClearBatch = lambda self:ClearEpoch(self, self.data)
        if not hasattr(Class, "AddBatchIndex"):
            Class.ClearBatch = lambda self:AddBatchIndex(self, self.data)
        if not hasattr(Class, "AddEpochIndex"):
            Class.ClearBatch = lambda self:AddEpochIndex(self, self.data)

    else:
        raise Exception(MountLocation)
#SetEpochBatchMethodForModule(AbstractModuleForEpochBatchTrain)