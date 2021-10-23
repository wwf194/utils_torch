import random

import numpy as np
from numpy import select, unravel_index

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as mpl
from matplotlib import pyplot as plt

from utils_torch.attrs import *
import utils_torch

class RNNLIF(nn.Module):
    # Singel-Layer Recurrent Neural Network with Leaky Integrate-and-Fire Dynamics
    def __init__(self, param=None, data=None, **kw):
        super(RNNLIF, self).__init__()
        utils_torch.model.InitForModel(self, param, data, ClassPath="utils_torch.Modules.RNNLIF", **kw)
    def InitFromParam(self, IsLoad=False):
        utils_torch.model.InitFromParamForModel(self, IsLoad)
        param = self.param
        data = self.data
        cache = self.cache
        
        if cache.IsInit:
            utils_torch.AddLog("RNNLIF: Initializing...")
        else:
            utils_torch.AddLog("RNNLIF: Loading...")
        CheckAttrs(param, "Type", value="RNNLIF")

        Neurons = param.Neurons
        EnsureAttrs(Neurons.Recurrent, "IsExciInhi", value=True)
        if GetAttrs(Neurons.Recurrent.IsExciInhi):
            EnsureAttrs(Neurons, "Recurrent.Excitatory.Ratio", default=0.8)
        cache.NeuronNum = Neurons.Recurrent.Num

        self.SetIterationTime()

        self.BuildModules()
        self.InitModules()
        self.DoInitTasks()
        self.ParseRouters()

        if cache.IsInit:
            utils_torch.AddLog("RNNLIF: Initialized.")
        else:
            utils_torch.AddLog("RNNLIF: Loaded.")
    def GenerateZeroInitState(self, RefInput):
        data = self.data
        cache = self.cache
        BatchSize = RefInput.size(0)
        InitState = torch.zeros((BatchSize, cache.NeuronNum * 2), device=self.GetTensorLocation(), requires_grad=False)
        return InitState
    def SetIterationTime(self):
        param = self.param
        cache = self.cache
        EnsureAttrs(param, "Iteration.Time", default="FromInput")
        if isinstance(param.Iteration.Time, int):
            cache.IterationTime = param.Iteration.Time
            self.GetIterationTime = lambda:cache.IterationTime
        elif param.Iteration.Time in ["FromInput"]:
            pass
        else:
            raise Exception(param.Iteration.Time)

__MainClass__ = RNNLIF
utils_torch.model.SetMethodForModelClass(__MainClass__)