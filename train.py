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
    for EpochIndex in range(param.Epoch.Num):
        logger.SetLocal("Epoch", EpochIndex)
        utils_torch.AddLog("Epoch: %d"%EpochIndex)
        for BatchIndex in range(param.Batch.Num):
            logger.SetLocal("Batch", BatchIndex)
            utils_torch.AddLog("Batch: %d"%BatchIndex)
            utils_torch.CallGraph(Router, In=In)

def ProcessMNIST(dataset_dir, augment=True, batch_size=64):    
    transform = transforms.Compose(
    [transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=dataset_dir, transform=transform, train=True, download=False)
    testset = torchvision.datasets.MNIST(root=dataset_dir, transform=transform, train=False, download=False)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader

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
        correct_count+=(torch.max(outputs, 1)[1]==labels).sum().item()
        labels_count+=labels.size(0)
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



