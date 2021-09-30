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
    # param = utils_torch.parse.ParsePyObjDynamic(param, InPlace=False, **kw)
    Router = utils_torch.router.ParseRouterStaticAndDynamic(param.Batch.Internal, ObjRefList=[param.Batch.Internal], **kw)
    In = utils_torch.parse.ParsePyObjDynamic(param.Batch.Input, **kw)
    
    logger.SetLocal("EpochNum", param.Epoch.Num)
    logger.SetLocal("BatchNum", param.Batch.Num)
    
    for EpochIndex in range(param.Epoch.Num):
        logger.SetLocal("EpochIndex", EpochIndex)
        utils_torch.AddLog("Epoch: %d"%EpochIndex)
        for BatchIndex in range(param.Batch.Num):
            logger.SetLocal("BatchIndex", BatchIndex)
            utils_torch.AddLog("Batch: %d"%BatchIndex)
            utils_torch.CallGraph(Router, In=In)
            logger.PlotAllLogs(SaveDir=utils_torch.GetSaveDir() + "log/")
            continue

def PlotTrainCurve(records, EpochNum, BatchNum, Name="Train"):
    Xs = []
    Ys = []
    for record in records:
        EpochIndex = record["Epoch"]
        BatchIndex = record["Batch"]
        Xs.append(EpochIndex + BatchIndex / BatchNum)
        Ys.append(record["Value"])
    utils_torch.plot.PlotLineChart(None, Xs, Ys, Save=True, SavePath="./%s.png"%Name)

class GradientDescend:
    def __init__(self, param=None):
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
            utils_torch.GetDataLogger().AddLog("%s.ChangeRatio"%Name, utils_torch.model.CalculateWeightChangeRatio(Weight, WeightChange))
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

'''
class model(nn.Module):
    def __init__(self):
        super(net_model, self).__init__()
        self.w1 = torch.nn.Parameter(1e-3*torch.rand(784, hidden_layer_size, device=device))
        self.b1 = torch.nn.Parameter(torch.zeros(hidden_layer_size, device=device))
        self.r1 = torch.nn.Parameter(1e-3*torch.rand(hidden_layer_size, hidden_layer_size,device=device))

        self.w2 = torch.nn.Parameter(1e-3*torch.rand(512, 10, device=device))
        self.b2 = torch.nn.Parameter(torch.zeros(10,device=device))

        
        self.mlp = torch.nn.Linear(96, 96, bias=True)

        self.relu = torch.nn.ReLU()

    def forward(self, x): # inputs : [batchsize, input_dim, time_windows]
        #change shape
        x = inputs.view(-1, 28 * 28)
        batch_size=x.size(0)
        input_dim=x.size(1)
        h = torch.zeros(batch_size, hidden_layer_size, device=device)
        for step in range(iter_time):
            h=h.mm(self.r1)            
            h=x+h+b1
            h=self.relu(h)
        
        x=h.mm(w2)+b2
        return x
'''

def GetEpochFloat(EpochIndex, BatchIndex, BatchNum):
    return EpochIndex + BatchIndex / BatchNum * 1.0