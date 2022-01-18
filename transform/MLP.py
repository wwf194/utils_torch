import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import utils_torch
from utils_torch.attr import *

def Build(param):
    # to be implemented
    return
def load_model(param):
    return

from utils_torch.transform import AbstractTransformWithTensor
#from utils_torch.transform.__init__ import AbstractTransformWithTensor
class MLP(AbstractTransformWithTensor):
    #def __init__(self, param=None, data=None, **kw):
        # super(MLP, self).__init__()
        # self.InitModule(self, param, data, ClassPath="utils_torch.transform.MLP", **kw)
    def __init__(self, **kw):
        super().__init__(**kw)
        return
    def Build(self, IsLoad=False, LoadDir=None):
        self.BeforeBuild(IsLoad=IsLoad)
        param = self.param
        data = self.data
        cache = self.cache

        if cache.IsInit:
            EnsureAttr(param.Neurons.NonLinear, "ReLU")

        cache.LayerNum = param.Layers.Num
        if cache.IsInit:
            EnsureAttr(param.Layers.Bias, True)
            EnsureAttr(param.Layers.Type, "NonLinearLayer")
            EnsureAttr(param.Layers.Subtype, "f(Wx+b)")
            EnsureAttr(param.Layers.LinearOnLastLayer, True)
            if not HasAttr(param.Layers.Num):
                param.Layers.Num = len(param.Layers.Neurons.Num) - 1
            EnsureAttr(param.Layers.Num, len(param.Neurons.Num) - 1)

            self.SetNeuronsNum()
            for LayerIndex in range(param.Layers.Num):
                LayerParam = eval("param.Modules.Layer%d"%LayerIndex)
                #LayerParam = utils_torch.EmptyPyObj()
                EnsureAttr(LayerParam.Type,      param.Layers.Type)
                EnsureAttr(LayerParam.Subtype,   param.Layers.Subtype)
                EnsureAttr(LayerParam.NonLinear, param.Neurons.NonLinear)
                SetAttr(LayerParam.FullName,     param.FullName + "." + "Layer%d"%LayerIndex)
                SetAttr(LayerParam.Input.Num,    param.Layers.Neurons.Num[LayerIndex])
                SetAttr(LayerParam.Output.Num,   param.Layers.Neurons.Num[LayerIndex + 1])

                if LayerParam.Subtype in ["NonLinear"]:
                    LayerParam.Subtype = "NonLinearLayer" # To avoid confusion between a nonlinear layer and a nonlinear function                                                                                                                                                                                                                                                                                                  
                if LayerIndex == cache.LayerNum - 1:
                    if param.Layers.LinearOnLastLayer:
                        LayerParam.Type = "LinearLayer"
                        if param.Layers.Subtype in ["f(Wx+b)"]:
                            LayerParam.Subtype = "Wx+b"
                        elif param.Layers.Subtype in ["f(Wx)"]:
                            LayerParam.Subtype = "Wx"

                if HasAttrs(param, "Layers.Weight.Init"):
                    SetAttrs(LayerParam, "Weight.Init", param.Layers.Weight.Init)
                #SetAttrs(param, "Modules.Layer%d"%LayerIndex, value=LayerParam)

        self.BuildModules(IsLoad=IsLoad, LoadDir=LoadDir)
        self.InitModules(IsLoad=IsLoad)
        self.ParseRouters(IsLoad=IsLoad)

        cache.Layers = []
        for LayerIndex in range(param.Layers.Num):
            #LayerParam = GetAttrs(param, "Modules.Layer%d"%LayerIndex)
            #Layer = utils_torch.module.BuildModule(LayerParam, LoadDir=cache.LoadDir)
            #SetAttrs(cache, "Modules.Layer%d"%LayerIndex, Layer)
            cache.Layers.append(getattr(cache.Modules, "Layer%d"%LayerIndex))             
            #self.add_module("Layer%d"%LayerIndex, Layer)
        # for Layer in cache.Layers:
        #     Layer.Build(IsLoad=cache.IsLoad)
        return self
    def SetNeuronsNum(self):
        param = self.param
        NeuronsNum = param.Layers.Neurons.Num
        NeuronsNum[0] = param.Neurons.Input.Num
        NeuronsNum[-1] = param.Neurons.Output.Num
        Index = 0
        while Index < len(NeuronsNum):
            #if NeuronsNum[Index] in ["Auto", "auto"]:
            if not isinstance(NeuronsNum, int):
                Index2 = Index
                while not isinstance(NeuronsNum[Index2], int):
                    Index2 += 1
                NeuronsNumFloat = np.linspace(NeuronsNum[Index - 1], NeuronsNum[Index2], Index2 - Index + 1)[1:]
                Index0 = Index
                while Index < Index2:
                    NeuronsNum[Index] = round(NeuronsNumFloat[Index - Index0])
                    Index += 1
                Index += 1
            else:
                Index += 1 


__MainClass__ = MLP
# utils_torch.transform.SetMethodForTransformModule(__MainClass__)