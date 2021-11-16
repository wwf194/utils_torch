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
    def Train(self, TrainData, TrainParam, log):
        Dynamics = self.Dynamics
        input = TrainData.input
        outputTarget = TrainData.outputTarget

        # inputInit = input.inputInit
        # inputSeries = input.inputSeries
        # IterationTime = input.IterationTime
        outputs = Dynamics.Run(input)

        log.hiddenStates = outputs.hiddenStateSeries
        log.cellStates = outputs.cellStateSeries
        log.firingRates = outputs.firingRateSeries
        log.output = outputs.outputSeries
        log.outputTarget = outputTarget

        Dynamics.Optimize(outputTarget, outputs.output, TrainParam, log=log)
        return
        # "Train":{
        #     "In":["input", "outputTarget", "trainParam"],
        #     "Out":[],
        #     "Routings":[
        #         "input |--> &Split |--> inputInit, inputSeries, time",
        #         "inputInit, inputSeries, time |--> &Run |--> outputSeries, hiddenStateSeries, cellStateSeries, firingRateSeries",
        #         "hiddenStateSeries, cellStateSeries, firingRateSeries |--> &Merge |--> activity",
        #         "outputSeries, outputTarget, activity, trainParam |--> &Optimize",

        #         "hiddenStateSeries, Name=HiddenStates,  logger=DataTrain |--> &LogTimeVaryingActivity",
        #         "cellStateSeries,   Name=CellStates,    logger=DataTrain |--> &LogTimeVaryingActivity",
        #         "firingRateSeries,  Name=FiringRates,   logger=DataTrain |--> &LogTimeVaryingActivity",
        #         "outputSeries,      Name=Outputs,       logger=DataTrain |--> &LogTimeVaryingActivity",
        #         "outputTarget,      Name=OutputTargets, logger=DataTrain |--> &LogTimeVaryingActivity",
                
        #         // "firingRateSeries,  Name=FiringRates, logger=DataTrain |-->   &LogSpatialActivity",
        #     ]
        # },
    
    def Optimize(self, output, outputTarget, activity, trainParam):

        return
    def Iterate(self, hiddenState, cellState):
        # hiddenState: recurrent input from last time step
        Modules = self.Modules
        Modules.Recurrent(hiddenState, cellState, inputProcessed)

        Modules.LoghiddenState.Receive(hiddenState)
        Modules.LogCellState.Receive(cellState)
        Modules.LogFiringRate.Receive(firingRate)
    def Run(self, input, IterationTime):
        cache = self.cache
        Modules = self.Modules
        Dynamics = self.Dynamics
        if IterationTime is None:
            IterationTime = cache.IterationTime
        
        Modules.InputManager.Receive(input)
        initState = self.GenerateZeroInitState(RefInput=input)
        hiddenState, cellState = Modules.SplitHiddenAndCellState(initState)
        for TimeIndex in range(IterationTime):
            hiddenState, cellState = Dynamics.Iterate(hiddenState, cellState)

        outputSeries = Modules.OutputManager.Send()
        hiddenStateSeries = Modules.HiddenStates.Send()
        cellStateSeries = Modules.CellStates.Send()
        firingRateSeries = Modules.FiringRates.Send()
        return {
            "outputSeries": outputSeries,
            "hiddenStateSeries": hiddenStateSeries,
            "cellStateSeries": cellStateSeries,
            "firingRateSeries": firingRateSeries,
        }
        # "In":["input", "time"],
        # "Out":["outputSeries", "hiddenStateSeries", "cellStateSeries", "firingRateSeries"],
        # "Routings":[
        #     "input |--> &InputManager.Receive",
        #     "RefInput=%input |--> &*GenerateZeroInitState |--> state", // States start from zero
        #     "state |--> &SplitHiddenAndCellState |--> hiddenState, cellState",
        #     "hiddenState, cellState |--> &Iterate |--> hiddenState, cellState || repeat=%time",
        #     "&OutputManager.Send |--> outputSeries",
        #     "&HiddenStates.Send |--> hiddenStateSeries",
        #     "&CellStates.Send |--> cellStateSeries",
        #     "&FiringRates.Send |--> firingRateSeries",
        # ]

__MainClass__ = RNNLIF
utils_torch.model.SetMethodForModelClass(__MainClass__)