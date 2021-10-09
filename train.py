import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import time
import os


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
from torch.optim.lr_scheduler import MultiStepLR,ReduceLROnPlateau

import torchvision
from torchvision.datasets import mnist
import torchvision.transforms as transforms

from collections import defaultdict

def Train(Args, **kw):
    if Args.Type in ["SupervisedLearning"]:
        if Args.SubType in ["EpochBatch"]:
            TrainEpochBatch(Args, **kw)
        else:
            raise Exception()
    else:
        raise Exception()

def TrainEpochBatch(param, **kw):
    kw["ObjCurrent"] = param
    logger = kw["Logger"]
    
    param = utils_torch.parse.ParsePyObjStatic(param, InPlace=True, **kw)
    RouterTrain = utils_torch.router.ParseRouterStaticAndDynamic(param.Batch.Train, ObjRefList=[param.Batch.Train], **kw)
    RouterTest = utils_torch.router.ParseRouterStaticAndDynamic(param.Batch.Test, ObjRefList=[param.Batch.Test], **kw)
    In = utils_torch.parse.ParsePyObjDynamic(param.Batch.Input, **kw)
    
    EpochNum = param.Epoch.Num
    BatchNum = param.Batch.Num

    logger.SetLocal("EpochNum", param.Epoch.Num)
    logger.SetLocal("BatchNum", param.Batch.Num)
    
    EpochIndex, BatchIndex = -1, BatchNum - 1
    logger.SetLocal("EpochIndex", EpochIndex)
    logger.SetLocal("BatchIndex", BatchIndex)
    utils_torch.CallGraph(RouterTest, In=In)
    AnalyzeAfterBatch(
        logger,
        EpochNum=EpochNum, EpochIndex=EpochIndex,
        BatchNum=BatchNum, BatchIndex=BatchIndex,
        **kw
    )

    for EpochIndex in range(EpochNum):
        logger.SetLocal("EpochIndex", EpochIndex)
        utils_torch.AddLog("Epoch: %d"%EpochIndex)
        for BatchIndex in range(BatchNum):
            logger.SetLocal("BatchIndex", BatchIndex)
            utils_torch.AddLog("Batch: %d"%BatchIndex)
            utils_torch.SetSaveDir(
                utils_torch.GetSaveDir() + "SavedModel/Epoch%d-Batch%d/"%(EpochIndex, BatchIndex),
                Type="Obj"
            )
            utils_torch.CallGraph(RouterTrain, In=In)
            # logger.PlotLogOfGivenType("WeightChangeRatio", PlotType="LineChart", 
            #     SaveDir=utils_torch.GetSaveDir() + "log/WeightChange")    
            if BatchIndex % 10 == 0:
                AnalyzeAfterBatch(
                    logger,
                    EpochNum=EpochNum, EpochIndex=EpochIndex,
                    BatchNum=BatchNum, BatchIndex=BatchIndex,
                    **kw
                )

def AnalyzeAfterBatch(logger, **kw):
    EpochIndex = kw["EpochIndex"]
    BatchIndex = kw["BatchIndex"]

    utils_torch.DoTasks("&^param.task.Save", **kw)
    utils_torch.DoTasks("&^param.task.Load", **kw)

    utils_torch.analysis.AnalyzeTrajectory(
        utils_torch.GetArgsGlobal().object.agent,
        utils_torch.GetArgsGlobal().object.world,
        logger.GetLog("agent.model.Outputs")["Value"],
        logger.GetLog("agent.model.OutputTargets")["Value"],
        SaveDir = utils_torch.GetSaveDir() + "Trajectory/",
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

    
def ParseTrainParamEpochBatch(param):
    EnsureAttrs(param, "Nesterov", value=False)
    EnsureAttrs(param, "Dampening", value=0.0)
    EnsureAttrs(param, "Momentum", value=0.0)

class GradientDescend:
    def __init__(self, param=None, data=None, **kw):
        self.cache = utils_torch.EmptyPyObj()
        self.cache.LastUpdateInfo = defaultdict(lambda:{})
    def InitFromParam(self):
        return
    def __call__(self, weights, param, ClearGrad=True, WarnNoneGrad=True):
        cache = self.cache
        for Name, Weight in weights.items():
            if Weight.grad is None:
                if WarnNoneGrad:
                    utils_torch.AddWarning("%s.grad is None."%Name)
                continue
            WeightChange = Weight.grad.data
            if param.WeightDecay != 0:
                #WeightChange.add_(param.WeightDecay, Weight.data)
                WeightChange.add_(Weight.data, alpha=param.WeightDecay,)
            if param.Momentum != 0:
                LastUpdateInfo = cache.LastUpdateInfo[Weight]
                if 'dW' not in LastUpdateInfo:
                    WeightChangeMomentum = LastUpdateInfo['dW'] = torch.clone(WeightChange).detach()
                else:
                    WeightChangeMomentum = LastUpdateInfo['dW']
                    #WeightChangeMomentum.mul_(param.Momentum).add_(1 - param.Dampening, WeightChange)
                    WeightChangeMomentum.mul_(param.Momentum).add_(WeightChange, alpha=1 - param.Dampening, )
                if param.Nesterov:
                    WeightChange = WeightChange.add(param.Momentum, alpha=WeightChangeMomentum)
                else:
                    WeightChange = WeightChangeMomentum
            #Weight.data.add_(-param.LearningRate, WeightChange)

            # if param.LimitWeightChangeRatio:
            #     RatioMax = param.WeightChangeRatioMax
            #     1.0 * torch.where(Weight == 0.0)
            # else:
            utils_torch.GetDataLogger().AddLog("%s.ChangeRatio"%Name,
                utils_torch.model.CalculateWeightChangeRatio(Weight, WeightChange),
                Type="WeightChangeRatio"
            )
            Weight.data.add_(WeightChange, alpha=-param.LearningRate)
            if ClearGrad:
                Weight.grad.detach_()
                Weight.grad.zero_()
        return
        # F.sgd(params: List[Tensor],
        #     d_p_list: List[Tensor],
        #     momentum_buffer_list: List[Optional[Tensor]],
        #     *,
        #     weight_decay: float,
        #     momentum: float,
        #     lr: float,
        #     dampening: float,
        #     nesterov: bool)

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