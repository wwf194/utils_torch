import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import GetAttrs, SetAttrs, HasAttrs, EnsureAttrs

def InitFromParam(param):
    # to be implemented
    return
def load_model(param):
    return

from utils_torch.module.AbstractModules import AbstractModuleWithTensor
#from utils_torch.transform.__init__ import AbstractModuleWithTensor
class MLP(AbstractModuleWithTensor):
    def __init__(self, param=None, data=None, **kw):
        super(MLP, self).__init__()
        utils_torch.transform.InitForModule(self, param, data, ClassPath="utils_torch.transform.MLP", **kw)
    def InitFromParam(self, IsLoad=False):
        utils_torch.transform.InitFromParamForModule(self, IsLoad)
        param = self.param
        data = self.data
        cache = self.cache

        cache.Modules = utils_torch.EmptyPyObj()
        if cache.IsInit:
            EnsureAttrs(param, "Init.Method", default="FromNeuronNum")
            EnsureAttrs(param, "NonLinear", default="ReLU")
            EnsureAttrs(param.Layers, "Bias", default="True")
            EnsureAttrs(param.Layers, "Type", default="f(Wx+b)")
        
        cache.LayerNum = param.Layers.Num
        if cache.IsInit:
            if param.Init.Method in ["FromNeuronNum"]:
                EnsureAttrs(param.Layers, "Num", default=len(param.Neurons.Num) - 1)
                EnsureAttrs(param.Layers, "LinearOnLastLayer", default=True)
                EnsureAttrs(param.Layers, "SubType", default="f(Wx+b)")
                EnsureAttrs(param.Layers, "Type", default="NonLinearLayer")

                for LayerIndex in range(param.Layers.Num):
                    LayerParam = utils_torch.EmptyPyObj()
                    SetAttrs(LayerParam, "Type", param.Layers.Type)
                    SetAttrs(LayerParam, "SubType", param.Layers.SubType)
                    #SetAttrs(LayerParam, "Bias", param.Layers.Bias)
                    SetAttrs(LayerParam, "Input.Num", value=param.Neurons.Num[LayerIndex])
                    SetAttrs(LayerParam, "Output.Num", value=param.Neurons.Num[LayerIndex + 1])
                    SetAttrs(LayerParam, "NonLinear", value=param.NonLinear)
                    SetAttrs(LayerParam, "FullName", value=param.FullName + "." + "Layer%d"%LayerIndex)
                    
                    if LayerIndex == cache.LayerNum - 1:
                        if param.Layers.LinearOnLastLayer:
                            SetAttrs(LayerParam, "Type", "LinearLayer")
                            if param.Layers.SubType in ["f(Wx+b)"]:
                                LayerParam.SubType = "Wx+b"
                            elif param.Layers.SubType in ["f(Wx)"]:
                                LayerParam.SubType = "Wx"

                    if HasAttrs(param, "Layers.Weight.Init"):
                        SetAttrs(LayerParam, "Weight.Init", param.Layers.Weight.Init)
                    SetAttrs(param, "Modules.Layer%d"%LayerIndex, value=LayerParam)
            else:
                raise Exception()
        
        cache.Layers = []
        for LayerIndex in range(param.Layers.Num):
            LayerParam = GetAttrs(param, "Modules.Layer%d"%LayerIndex)
            Layer = utils_torch.transform.BuildModule(LayerParam, LoadDir=cache.LoadDir)
            SetAttrs(cache, "Modules.Layer%d"%LayerIndex, Layer)
            cache.Layers.append(Layer)             
            self.add_module("Layer%d"%LayerIndex, Layer)
        for Layer in cache.Layers:
            Layer.InitFromParam(IsLoad=cache.IsLoad)
        
    def forward(self, Input):
        cache = self.cache
        States = {}
        States["0"] = Input
        for LayerIndex, Layer in enumerate(cache.Layers):
            Output = Layer.forward(States[str(LayerIndex)])
            States[str(LayerIndex + 1)] =Output
        return [
            States[str(cache.LayerNum)]
        ]

__MainClass__ = MLP
# utils_torch.transform.SetMethodForModuleClass(__MainClass__)