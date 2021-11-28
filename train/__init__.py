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

def Probability2MostProbableIndex(Probability):
    # Probability: [BatchSize, ClassNum]
    #Max, MaxIndices = torch.max(Probability, dim=1)
    #return utils_torch.TorchTensor2NpArray(MaxIndices) # [BatchSize]
    return torch.argmax(Probability, axis=1)

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
from utils_torch.train.Trainer import TrainerForEpochBatchTrain