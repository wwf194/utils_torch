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
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

# def Train(Args, **kw):
#     if Args.Type in ["SupervisedLearning"]:
#         if Args.SubType in ["EpochBatch"]:
#             TrainEpochBatch(Args, **kw)
#         else:
#             raise Exception()
#     else:
#         raise Exception()

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

class TrainerForEpochBatchTrain:
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
        # data.log = utils_torch.EmptyPyObj()
        # data.log.Train = utils_torch.EmptyPyObj()
        # data.log.Test = utils_torch.EmptyPyObj()
        self.BuildModules()
        self.InitModules()
        self.ParseRouters()
        self.ClearEpoch()
        self.ClearBatch()
        self.RegisterCheckPoint()
    def RegisterCheckPoint(self):
        cache = self.cache
        cache.CheckPointList = []
        for Name, Module in ListAttrsAndValues(cache.Modules, Exceptions=["__ResolveRef__", "__Entry__"]):
            if hasattr(Module, "IsCheckPoint") and Module.IsCheckPoint is True:
                self.cache.CheckPointList.append(Module)
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
            #"log": cache.LogTrain,
            "EpochNum": cache.EpochNum,
            "BatchNum": cache.BatchNum,
            "EpochIndex": cache.EpochIndex,
            "BatchIndex": cache.BatchIndex,
        })
    def __call__(self):
        utils_torch.CallGraph(self.Dynamics.Main)
    def ReportEpochBatch(self):
        cache = self.cache
        utils_torch.AddLog("Epoch%d-Batch%d"%(cache.EpochIndex, cache.BatchIndex))

utils_torch.module.SetMethodForNonModelClass(TrainerForEpochBatchTrain)
utils_torch.module.SetEpochBatchMethodForModule(TrainerForEpochBatchTrain)
                
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

def ParseOptimizeParamEpochBatch(param):
    EnsureAttrs(param, "Nesterov", value=False)
    EnsureAttrs(param, "Dampening", value=0.0)
    EnsureAttrs(param, "Momentum", value=0.0)

class CheckPointForEpochBatchTraining:
    def __init__(self, param, **kw):
        utils_torch.module.InitForNonModel(self, param, ClassPath="utils_torch.Train.CheckPointForEpochBatchTraining", **kw)
    def InitFromParam(self, IsLoad):
        utils_torch.module.InitFromParamForNonModel(self, IsLoad)
        # Intervals are calculated in batches, not epochs.
        param = self.param
        cache = self.cache
        EnsureAttrs(param, "CalculateCheckPointMode", default="Online")

        if param.CalculateCheckPointMode in ["Static"]: # For cases where EpochNum and BatchNum is known before training.
            assert HasAttrs(param, "Epoch.Num")
            assert HasAttrs(param, "Batch.Num")
            EnsureAttrs(param, "Interval.IncreaseCoefficient", value=1.5)
            EnsureAttrs(param, "Interval.Start", value=10)
            EnsureAttrs(param, "Interval.Max", value=10 * param.Batch.Num)
            cache.CheckPointList = self.CalculateCheckPointList(param)
            cache.BatchIndex = -1
            cache.CheckPointNextIndex = 0
            cache.CheckPointNext = self.CheckPointList[self.CheckPointNextIndex]
            self.AddBatchAndReturnIsCheckPoint = self.AddBatchAndReturnIsCheckPointStatic
        elif param.CalculateCheckPointMode in ["Online"]:
            EnsureAttrs(param, "Interval.IncreaseCoefficient", value=1.5)
            EnsureAttrs(param, "Interval.Start", value=10)
            EnsureAttrs(param, "Interval.Max", value=10000)
            cache.BatchIndex = -1
            cache.IntervalCurrent = param.Interval.Start
            cache.IntervalIndex = 0
            cache.IntervalMax = param.Interval.Max
            cache.IntervalIncreaseCoefficient = param.Interval.IncreaseCoefficient
            self.AddBatchAndReturnIsCheckPoint = self.AddBatchAndReturnIsCheckPointOnline
        elif param.CalculateCheckPointMode in ["Always", "EveryBatch"]:
            cache.BatchIndex = -1
            cache.IntervalIndex = 0
            self.AddBatchAndReturnIsCheckPoint = self.AddBatchAndReturnIsCheckPointAlwaysTrue
        else:
            raise Exception(param.CalculateCheckPointMode)
        
        EnsureAttrs(param, "Method", default="&#utils_torch.functions.NullFunction")
        cache.Method = utils_torch.parse.ResolveStr(
            param.Method,
            ObjCurrent=self.param,
            ObjRoot=utils_torch.GetGlobalParam()
        )
    def CalculateCheckPointList(param):
        BatchNumTotal = param.Epoch.Num * param.Batch.Num
        CheckPointBatchIndices = []
        BatchIndexTotal = 0
        CheckPointBatchIndices.append(BatchIndexTotal)
        Interval = param.Interval.Start
        while BatchIndexTotal < BatchNumTotal:
            BatchIndexTotal += round(Interval)
            CheckPointBatchIndices.append(BatchIndexTotal)
            Interval *= param.Interval.IncreaseCoefficient
            if Interval > param.Interval.Max:
                Interval =param.Interval.Max
        return CheckPointBatchIndices
    def AddBatchAndReturnIsCheckPointStatic(self, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        IsCheckPoint = False
        if cache.BatchIndex >= self.CheckPointNext:
            IsCheckPoint = True
            cache.CheckPointNextIndex += 1
            cache.CheckPointNext = cache.CheckPointList[self.CheckPointNextIndex]
        return IsCheckPoint
    def AddBatchAndReturnIsCheckPointOnline(self, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        cache.IntervalIndex += 1
        if cache.IntervalIndex >= cache.IntervalCurrent:
            IsCheckPoint = True
            cache.IntervalCurrent = round(cache.IntervalCurrent * cache.IntervalIncreaseCoefficient)
            if cache.IntervalCurrent > cache.IntervalMax:
                cache.IntervalCurrent = cache.IntervalMax
            cache.IntervalIndex = 0
        else:
            IsCheckPoint = False
        return IsCheckPoint
    def AddBatchAndReturnIsCheckPointAlwaysTrue(self, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        return True
    def GetMethod(self):
        return self.cache.Method
CheckPointForEpochBatchTraining.IsCheckPoint = True
utils_torch.module.SetEpochBatchMethodForModule(CheckPointForEpochBatchTraining)

def ClearGrad(weights):
    for name, weight in weights.items():
        if weight.grad is not None:
                weight.grad.detach_()
                weight.grad.zero_()

def evaluate(net, testloader, criterion, scheduler, augment, device):
    net.eval()
    count=0
    labels_count=0
    correct_count=0
    labels_count=0
    loss_total=0.0
    #torch.cuda.empty_cache()
    for data in testloader:
        #print("\r","progress:%d/%d "%(count,len(testloader)), end="", flush=True)
        count=count+1
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)
        if(augment==True):
            bs, ncrops, c, h, w = inputs.size()
            outputs = net(inputs.view(-1, c, h, w))
            outputs = outputs.view(bs, ncrops, -1).mean(1)
        else:
            outputs = net(inputs) 
            outputs = outputs.to(device)
        loss_total += criterion(outputs, labels).item()
        correct_count += (torch.max(outputs, 1)[1]==labels).sum().item()
        labels_count += labels.size(0)
    #print("\n")
    val_loss=loss_total/count
    val_acc=correct_count/labels_count
    net.train()
    return val_loss, val_acc

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

def CmpEpochBachData(data1, data2):
    EpochIndex1 = data1.EpochIndex
    BatchIndex1 = data1.BatchIndex
    EpochIndex2 = data2.EpochIndex
    BatchIndex2 = data2.BatchIndex
    return CmpEpochBatch(EpochIndex1, BatchIndex1, EpochIndex2, BatchIndex2)

def CmpEpochBachDict(Dict1, Dict2):
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