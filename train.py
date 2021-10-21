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

def Train(Args, **kw):
    if Args.Type in ["SupervisedLearning"]:
        if Args.SubType in ["EpochBatch"]:
            TrainEpochBatch(Args, **kw)
        else:
            raise Exception()
    else:
        raise Exception()

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

class TrainerForEpochBatchTraining:
    def __init__(self, param, **kw):
        utils_torch.model.InitForNonModel(self, param, **kw)
    def InitFromParam(self, IsLoad=True):
        param = self.param
        cache = self.cache
        utils_torch.model.InitFromParamForModel(self, IsLoad)
        self.BuildModules()
        self.ParseRouters()
        self.ClearEpoch()
        self.ClearBatch()
    def SetBatchNum(self, BatchNum):
        self.cache.BatchNum = BatchNum
    def SetEpochNum(self, EpochNum):
        self.cache.EpochNum = EpochNum
    def ClearBatch(self):
        self.cache.BatchIndex = 0
    def ClearEpoch(self):
        self.cache.EpochIndex = 0
    def AddBatch(self):
        self.cache.BatchIndex += 1
    def AddEpoch(self):
        self.cache.EpochIndex += 1
    def NotifyEpochIndex(self):
        cache = self.cache
        for Obj in self.cache.NotifyEpochBatchList:
            Obj.NotifyEpochIndex(cache.EpochIndex)
    def NotifyBatchIndex(self):
        cache = self.cache
        for Obj in self.cache.NotifyEpochBatchList:
            Obj.NotifyBatchIndex(cache.BatchIndex)
    def NotifyEpochNum(self):
        cache = self.cache
        for Obj in self.cache.NotifyEpochBatchList:
            Obj.NotifyEpochNum(cache.EpochNum)
    def NotifyBatchNum(self):
        cache = self.cache
        for Obj in self.cache.NotifyEpochBatchList:
            Obj.NotifyBatchNum(cache.BatchNum)
    def __call__(self):
        utils_torch.CallGraph(self.Dynamics.Main)

utils_torch.model.SetMethodForNonModelClass(TrainerForEpochBatchTraining)

def TrainEpochBatch(param, **kw):
    kw["ObjCurrent"] = param
    logger = kw["Logger"]
    
    param = utils_torch.parse.ParsePyObjStatic(param, InPlace=True, **kw)
    Routers = ParseRoutersFromTrainParam(param, **kw)
    NofityEpochBatchList = utils_torch.parse.ParsePyObjDynamic(
        param.NotifyEpochBatchList, InPlace=False, **kw
    )

    EpochNum, BatchNum = param.Epoch.Num, param.Batch.Num
    NotifyEpochNum(NofityEpochBatchList, EpochNum)
    NotifyBatchNum(NofityEpochBatchList, BatchNum)
    
    EpochIndex, BatchIndex = -1, BatchNum - 1
    NotifyEpochIndex(NofityEpochBatchList, EpochIndex)
    NotifyBatchIndex(NofityEpochBatchList, BatchIndex)

    utils_torch.CallGraph(Routers.Test, In=Routers.In)

    SaveAndLoad(EpochIndex, BatchIndex, **kw)
    Routers = ParseRoutersFromTrainParam(param, **kw)

    AnalyzeAfterBatch(
        utils_torch.GetLogger("DataTest"),
        Routers=Routers, param=param,
        EpochNum=EpochNum, EpochIndex=EpochIndex,
        BatchNum=BatchNum, BatchIndex=BatchIndex,
        **kw
    )

    CheckPointSave = CheckPoint()
    CheckPointAnalyze = CheckPoint()
    CheckPointSave.SetCheckPoint(EpochNum, BatchNum, IntervalStart=20)
    CheckPointAnalyze.SetCheckPoint(EpochNum, BatchNum, IntervalStart=20)


    for EpochIndex in range(EpochNum):
        NotifyEpochIndex(NofityEpochBatchList, EpochIndex)
        #utils_torch.AddLog("Epoch: %d"%EpochIndex)
        for BatchIndex in range(BatchNum):
            NotifyBatchIndex(NofityEpochBatchList, BatchIndex)
            utils_torch.AddLog("Epoch%d-Batch%d"%(EpochIndex, BatchIndex))
            utils_torch.CallGraph(Routers.Train, In=Routers.In) 
            
            # Save and Reload.
            if CheckPointSave.AddBatchAndReturnIsCheckPoint():
                SaveAndLoad(EpochIndex, BatchIndex, **kw)
                Routers = ParseRoutersFromTrainParam(param, **kw)
            
            # Do analysis
            if CheckPointAnalyze.AddBatchAndReturnIsCheckPoint():
                AnalyzeAfterBatch(
                    utils_torch.GetLogger("Data"),
                    Routers=Routers, param=param,
                    EpochNum=EpochNum, EpochIndex=EpochIndex,
                    BatchNum=BatchNum, BatchIndex=BatchIndex,
                    **kw
                )
                
def ParseRoutersFromTrainParam(param, **kw):
    Routers = utils_torch.PyObj()
    for Name, RouterParam in ListAttrsAndValues(param.Batch.Routers):
        Router = utils_torch.router.ParseRouterStaticAndDynamic(RouterParam, ObjRefList=[RouterParam, param], **kw)
        setattr(Routers, Name, Router)
    return Routers

def SaveAndLoad(EpochIndex, BatchIndex, **kw):
    SaveDir = utils_torch.SetSubSaveDirEpochBatch("SavedModel", EpochIndex, BatchIndex)
    utils_torch.DoTasks(
            "&^param.task.Save", 
            In={"SaveDir": SaveDir},
            **kw
        )
    utils_torch.DoTasks(
        "&^param.task.Load",
        In={"SaveDir": SaveDir}, 
        **kw
    )

def SetSaveDirForSavedModel(EpochIndex, BatchIndex):
    SaveDirForSavedModel = utils_torch.GetMainSaveDir() + "SavedModel/" + "Epoch%d-Batch%d/"%(EpochIndex, BatchIndex)
    utils_torch.SetSubSaveDir(SaveDirForSavedModel, Type="Obj")

def AnalyzeSptialActivity(param, **kw):
    Routers = kw.get("Routers")
    EpochIndex, BatchIndex = kw.get("EpochIndex"), kw.get("BatchIndex")
    logger = utils_torch.GetLogger("DataTest")
    NeuronNum = utils_torch.parse.ResolveStr(param.SpatialActivityMap.NeuronNum, **kw)
    BoundaryBox = utils_torch.parse.ResolveStr(param.SpatialActivityMap.BoundaryBox, **kw)
    Arena = utils_torch.parse.ResolveStr(param.Arena, **kw)
    SpatialActivity = utils_torch.ExternalMethods.InitSpatialActivity(
        BoundaryBox = BoundaryBox,
        Resolution = param.SpatialActivityMap.Resolution,
        NeuronNum = NeuronNum
    )
    for _BatchIndex in range(param.Batch.Num):
        utils_torch.AddLog("Analyzing Spatial Activity. Batch%d"%_BatchIndex)
        utils_torch.CallGraph(Routers.Test, In=Routers.In)
        activity = logger.GetLogByName("agent.model.FiringRates")["Value"]
        XYsTruth = logger.GetLogByName("agent.model.Outputs")["Value"]
        XYsPredicted = logger.GetLogByName("agent.model.OutputTargets")["Value"]
        # Use either XYsTruth or XYsPredicted for spatial activity map plotting.
        utils_torch.ExternalMethods.LogSpatialActivity(SpatialActivity, activity, XYsPredicted)
    utils_torch.ExternalMethods.CalculateSpatialActivity(SpatialActivity)
    utils_torch.ExternalMethods.PlotSpatialActivity(
        SpatialActivity, Arena, NeuronName="Hidden Neuron",
        SaveDir = utils_torch.GetMainSaveDir() + "SpatialActivityMap/" + "Epoch%d-Batch%d/"%(EpochIndex, BatchIndex),
        SaveName = "agent.model.FiringRates"
    )

def AnalyzeAfterBatch(logger, **kw):
    EpochIndex = kw["EpochIndex"]
    BatchIndex = kw["BatchIndex"]
    Routers = kw.get("Routers")
    param = kw.get("param")

    _kw = dict(kw)
    _kw["param"] = param.TestForSpatialActivityAnalysis
    AnalyzeSptialActivity(
        **_kw
    )
    utils_torch.analysis.AnalyzeTrajectory(
        utils_torch.GetGlobalParam().object.agent,
        utils_torch.GetGlobalParam().object.world,
        logger.GetLog("agent.model.Outputs")["Value"],
        logger.GetLog("agent.model.OutputTargets")["Value"],
        SaveDir = utils_torch.GetMainSaveDir() + "Trajectory/",
        SaveName = "Trajectory-Truth-Predicted-Epoch%d-Batch%d"%(EpochIndex, BatchIndex)
    )

    utils_torch.analysis.AnalyzeLossEpochBatch(
        Logs=logger.GetLogOfType("Loss"), **kw
    )
    utils_torch.analysis.AnalyzeTimeVaryingActivitiesEpochBatch(
        Logs=logger.GetLogOfType("TimeVaryingActivity"),
    )
    utils_torch.analysis.AnalyzeWeightsEpochBatch(
        Logs=logger.GetLogOfType("Weight"),
    )
    utils_torch.analysis.AnalyzeWeightStatAlongTrainingEpochBatch(
        Logs=logger.GetLogOfType("Weight-Stat"), **kw
    )

    if logger.GetLogByName("MinusGrad") is not None:
        utils_torch.analysis.AnalyzeResponseSimilarityAndWeightUpdateCorrelation(
            ResponseA=logger.GetLogByName("agent.model.FiringRates")["Value"],
            ResponseB=logger.GetLogByName("agent.model.FiringRates")["Value"],
            WeightUpdate=logger.GetLogByName("MinusGrad")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
            Weight = logger.GetLogByName("Weight")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
            SaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis/",
            SaveName = "Epoch%d-Batch%d-Recurrent.FiringRate2RecurrentInput.Weight"%(EpochIndex, BatchIndex),
        )

    return

def CallGraphEpochBatch(router, In, logger, EpochIndex, BatchIndex):
    logger.SetLocal("EpochIndex", EpochIndex)
    logger.SetLocal("BatchIndex", BatchIndex)
    utils_torch.AddLog("Epoch%d-Batch%d"%(EpochIndex, BatchIndex))
    utils_torch.CallGraph(router, In=In) 

def AddAnalysis():
    # Do supplementary analysis for all saved models under main save directory.
    kw = {
        "ObjRoot": utils_torch.GetGlobalParam()
    }
    SaveDirs = utils_torch.GetAllSubSaveDirsEpochBatch("SavedModel")
    for SaveDir in SaveDirs:
        EpochIndex, BatchIndex = ParseEpochBatchFromStr(SaveDir)
        logger = utils_torch.GetLogger("DataTest")
        logger.NotifyEpochIndex(EpochIndex)
        logger.NotifyBatchIndex(BatchIndex)

        utils_torch.DoTasks(
            "&^param.task.Load",
            In={"SaveDir": SaveDir}, 
            **kw
        )

        param = utils_torch.parse.ResolveStr("&^param.task.Train", **kw)

        kw["ObjCurrent"] = param
        param = utils_torch.parse.ParsePyObjStatic(param, InPlace=True, **kw)
        Routers = ParseRoutersFromTrainParam(param, **kw)
        # AnalyzeSptialActivity(
        #     param=param.TestForSpatialActivityAnalysis,
        #     Routers=Routers,
        #     EpochIndex=EpochIndex, BatchIndex=BatchIndex,
        #     ObjCurrent=param, ObjRoot=utils_torch.GetGlobalParam()
        # )
        CallGraphEpochBatch(Routers.Test, Routers.In, logger, EpochIndex, BatchIndex)

        # utils_torch.analysis.AnalyzeTrajectory(
        #     utils_torch.GetGlobalParam().object.agent,
        #     utils_torch.GetGlobalParam().object.world,
        #     logger.GetLog("agent.model.Outputs")["Value"],
        #     logger.GetLog("agent.model.OutputTargets")["Value"],
        #     SaveDir = utils_torch.GetMainSaveDir() + "Trajectory/",
        #     SaveName = "Trajectory-Truth-Predicted-Epoch%d-Batch%d"%(EpochIndex, BatchIndex),
        #     PlotNum = 1
        # )
        utils_torch.analysis.AnalyzeResponseSimilarityAndWeightUpdateCorrelation(
            ResponseA=logger.GetLogByName("agent.model.FiringRates")["Value"],
            ResponseB=logger.GetLogByName("agent.model.FiringRates")["Value"],
            WeightUpdate=logger.GetLogByName("MinusGrad")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
            Weight = logger.GetLogByName("Weight")["Value"]["Recurrent.FiringRate2RecurrentInput.Weight"],
            SaveDir = utils_torch.GetMainSaveDir() + "Hebb-Analysis/",
            SaveName = "Epoch%d-Batch%d-Recurrent.FiringRate2RecurrentInput.Weight"%(EpochIndex, BatchIndex),
        )

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

def BuildCheckPoint(param):
    checkPoint = CheckPoint(param)
    checkPoint.InitFromParam()
    return checkPoint

class CheckPointForEpochBatchTraining:
    def __init__(self, param):
        self.param = param
        return
    def InitFromParam(self):
        # Intervals are calculated in batches, not epochs.
        param = self.param
        assert HasAttrs(param, "Epoch.Num")
        assert HasAttrs(param, "Batch.Num")
        EnsureAttrs(param, "Interval.IncreaseCoefficient", value=1.5)
        EnsureAttrs(param, "Interval.Start", value=10)
        EnsureAttrs(param, "Interval.Max", value=10 * param.BatchNum)
        self.CheckPointList = GetCheckPointListEpochBatch(param)
        self.BatchIndex = -1
        self.CheckPointNextIndex = 0
        self.CheckPointNext = self.CheckPointList[self.CheckPointNextIndex]
    def AddBatchAndReturnIsCheckPoint(self):
        self.BatchIndex += 1
        IsCheckPoint = False
        if self.BatchIndex >= self.CheckPointNext:
            IsCheckPoint = True
            self.CheckPointNextIndex += 1
            self.CheckPointNext = self.CheckPointList[self.CheckPointNextIndex]
        return IsCheckPoint
    
def GetCheckPointListEpochBatch(param):
    BatchNumTotal = param.EpochNum * param.BatchNum
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

def evaluate_iter(net, testloader, criterion, scheduler, augment, device):
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
            outputs, act = net(inputs)
            #outputs = list(map(lambda x:x.to(device), outputs))  
        loss_total += net.Getloss(inputs, labels).item()
        correct_count+=(torch.max(outputs[-1], 1)[1]==labels).sum().item()
        labels_count+=labels.size(0)
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

def Probability2MostProbableClassIndex(Probabilities):
    # Probabilities: [BatchSize, ClassNum]
    Max, MaxIndices = torch.max(Probabilities, dim=1)
    return utils_torch.TorchTensor2NpArray(MaxIndices) # [BatchSize]

def InitAccuracy():
    Accuracy = utils_torch.PyObj()
    Accuracy.Num = 0
    Accuracy.NumCorrect = 0
    return Accuracy

def ResetAccuracy(Accuracy):
    Accuracy.Num = 0
    Accuracy.NumCorrect = 0
    return Accuracy

def CalculateAccuracy(Accuracy):
    Accuracy.RatioCorrect = 1.0 * Accuracy.NumCorrect / Accuracy.Num
    return Accuracy

def LogAccuracyForSingleClassPrediction(Accuracy, Output, OutputTarget):
    # Output: np.ndarray. Predicted class indices in shape of [BatchNum]
    # OutputTarget: np.ndarray. Ground Truth class indices in shape of [BatchNum]
    SampleNum = Output.shape[0]
    CorrectNum = np.sum(Output==OutputTarget)
    Accuracy.Num += SampleNum
    Accuracy.NumCorrect += CorrectNum