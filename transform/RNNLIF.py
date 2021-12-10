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

from utils_torch.transform import AbstractTransformWithTensor
class RNNLIF(AbstractTransformWithTensor):
    # Singel-Layer Recurrent Neural Network with Leaky Integrate-and-Fire Dynamics
    # def __init__(self, param=None, data=None, **kw):
    #     super(RNNLIF, self).__init__()
    #     self.InitModule(self, param, data, ClassPath="utils_torch.transform.RNNLIF", **kw)
    def __init__(self, **kw):
        super().__init__(**kw)
        return
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        param = self.param
        data = self.data
        cache = self.cache
        
        if cache.IsInit:
            utils_torch.AddLog("RNNLIF: Initializing...")
        else:
            utils_torch.AddLog("RNNLIF: Loading...")
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


        return self
    def SetNeuronsNum(self, InputNum, OutputNum, HiddenNeuronsNum=None):
        param = self.param
        SetAttrs(param, "Neurons.Input.Num", value=InputNum)
        SetAttrs(param, "Neurons.Output.Num", value=OutputNum)
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
    def Train(self, TrainData, OptimizeParam, log):
        Dynamics = self.Dynamics
        input = TrainData.input
        outputTarget = TrainData.outputTarget

        # inputInit = input.inputInit
        # inputSeries = input.inputSeries
        # IterationTime = input.IterationTime
        outputs = Dynamics.Run(input)

        log.recurrentInput = outputs.recurrentInputSeries
        log.membranePotential = outputs.membranePotentialSeries
        log.firingRates = outputs.firingRateSeries
        log.output = outputs.outputSeries
        log.outputTarget = outputTarget

        Dynamics.Optimize(outputTarget, outputs.output, OptimizeParam, log=log)
        return
        # "Train":{
        #     "In":["input", "outputTarget", "OptimizeParam"],
        #     "Out":[],
        #     "Routings":[
        #         "input |--> &Split |--> inputInit, inputSeries, time",
        #         "inputInit, inputSeries, time |--> &Run |--> outputSeries, recurrentInputSeries, membranePotentialSeries, firingRateSeries",
        #         "recurrentInputSeries, membranePotentialSeries, firingRateSeries |--> &Merge |--> activity",
        #         "outputSeries, outputTarget, activity, OptimizeParam |--> &Optimize",

        #         "recurrentInputSeries, Name=HiddenStates,  logger=DataTrain |--> &LogActivityAlongTime",
        #         "membranePotentialSeries,   Name=MembranePotential,    logger=DataTrain |--> &LogActivityAlongTime",
        #         "firingRateSeries,  Name=FiringRates,   logger=DataTrain |--> &LogActivityAlongTime",
        #         "outputSeries,      Name=Outputs,       logger=DataTrain |--> &LogActivityAlongTime",
        #         "outputTarget,      Name=OutputTargets, logger=DataTrain |--> &LogActivityAlongTime",
                
        #          "firingRateSeries,  Name=FiringRates, logger=DataTrain |-->   &LogSpatialActivity",
        #     ]
        # },
    
    def Optimize(self, output, outputTarget, activity, OptimizeParam):
        return
    def Iterate(self, input, recurrentInput, membranePotential, log):
        # recurrentInput: recurrent input from last time step
        Modules = self.Modules
        return Modules.RecurrentLIFNeurons(recurrentInput, membranePotential, input)
    def Run(self, Input, IterationTime, log:utils_torch.log.LogAlongEpochBatchTrain):
        cache = self.cache
        Modules = self.Modules
        Dynamics = self.Dynamics
        if IterationTime is None:
            IterationTime = cache.IterationTime
        Modules.InputList.Receive(Input)
        initState = self.GenerateZeroInitState(RefInput=Input)
        #recurrentInput, membranePotential = Modules.SplitHiddenAndCellState(initState)
        
        recurrentInput = initState[:, cache.NeuronNum] # Initial recurrentInput
        membranePotential = initState[:, cache.NeuronNum] # Initial membranePotential

        Modules.InputHolder.Receive(Input)
        for TimeIndex in range(IterationTime):
            input = Modules.InputHolder.Send()        
            Modules.InputList.Receive(input)
            inputTransformed = Modules.TransformInput(input)
            Modules.InputTransformedList.Receive(inputTransformed)
            recurrentInput, membranePotential = Dynamics.Iterate(inputTransformed, recurrentInput, membranePotential, log=log)

        outputList = Modules.OutputList.Send()
        recurrentInputList = Modules.HiddenStateList.Send()
        membranePotentialList = Modules.MembranePotentialList.Send()
        firingRateList = Modules.FiringRateList.Send()
        inputList = Modules.InputList.Send()

        log.AddLogCache("OutputList", outputList, "ActivityAlongTime")
        log.AddLogCache("RecurrentInputList", recurrentInputList, "ActivityAlongTime")
        log.AddLogCache("MembranePotentialList", membranePotentialList, "ActivityAlongTime")
        log.AddLogCache("FiringRateList", firingRateList, "ActivityAlongTime")
        log.AddLogCache("InputList", inputList, "ActivityAlongTime")
        log.AddLogCache("InputTransformed", inputTransformed, "ActivityAlongTime")
        
        return outputList        
        # return {
        #     "outputList": outputList
        #     # "outputSeries": outputSeries,
        #     # "recurrentInputSeries": recurrentInputSeries,
        #     # "membranePotentialSeries": membranePotentialSeries,
        #     # "firingRateSeries": firingRateSeries,
        # }
        # {
        #     "In":["input", "time", "log"],
        #     "Out":["logOutput", "logRecurrentInput", "logMembranePotential", "logFiringRate"],
        #     "Routings":[
        #         "input |--> &InputCache.Receive",
        #         "RefInput=%input |--> &*GenerateZeroInitState |--> state",  States start from zero
        #         "state |--> &SplitHiddenAndMembranePotential |--> recurrentInput, membranePotential",
        #         "recurrentInput, membranePotential |--> &Iterate |--> recurrentInput, membranePotential || repeat=%time",
        #         "&LogOutput.Send            |--> logOutput",
        #         "&LogRecurrentInput.Send    |--> logRecurrentInput",
        #         "&LogMembranePotential.Send |--> logMembranePotential",
        #         "&LogFiringRate.Send        |--> logFiringRate",
        #     ]
        # },

__MainClass__ = RNNLIF
#utils_torch.transform.SetMethodForTransformModule(__MainClass__)