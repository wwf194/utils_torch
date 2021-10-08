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

class MLP(torch.nn.Module):
    def __init__(self, param=None, data=None, **kw):
        super(MLP, self).__init__()
        utils_torch.model.InitForModel(self, param, data, ClassPath="utils_torch.Models.MLP", **kw)
    def InitFromParam(self, param=None, IsLoad=False):
        if param is None:
            param = self.param
            data = self.data
            cache = self.cache
        else:
            self.param = param
        cache.IsLoad = IsLoad
        cache.IsInit = not cache.IsLoad

        cache.Modules = utils_torch.EmptyPyObj()
        if cache.IsInit:
            EnsureAttrs(param, "Init.Method", default="FromNeuronNum")
            EnsureAttrs(param, "NonLinear", default="ReLU")
            EnsureAttrs(param.Layers, "Bias", default="True")
            EnsureAttrs(param.Layers, "Type", default="f(Wx+b)")
        
        if cache.IsInit:
            if param.Init.Method in ["FromNeuronNum"]:
                EnsureAttrs(param.Layers, "Num", default=len(param.Neurons.Num) - 1)
                for LayerIndex in range(param.Layers.Num):
                    LayerParam = utils_torch.EmptyPyObj()
                    SetAttrs(LayerParam, "Type", "NonLinearLayer")
                    SetAttrs(LayerParam, "Subtype", param.Layers.Type)
                    SetAttrs(LayerParam, "Bias", param.Layers.Bias)
                    SetAttrs(LayerParam, "Input.Num", value=param.Neurons.Num[LayerIndex])
                    SetAttrs(LayerParam, "Output.Num", value=param.Neurons.Num[LayerIndex + 1])
                    SetAttrs(LayerParam, "NonLinear", value=param.NonLinear)
                    SetAttrs(LayerParam, "FullName", value=param.FullName + "." + "Layer%d"%LayerIndex)
                    SetAttrs(param, "Modules.Layer%d"%LayerIndex, value=LayerParam)
            else:
                raise Exception()
        
        cache.Layers = []
        for LayerIndex in range(param.Layers.Num):
            LayerParam = GetAttrs(param, "Modules.Layer%d"%LayerIndex)
            Layer = utils_torch.model.BuildModule(LayerParam, LoadDir=cache.LoadDir)
            SetAttrs(cache, "Modules.Layer%d"%LayerIndex, Layer)
            cache.Layers.append(Layer)             
            self.add_module("Layer%d"%LayerIndex, Layer)
        for Layer in cache.Layers:
            Layer.InitFromParam(IsLoad=cache.IsLoad)
        cache.LayerNum = len(cache.Layers)
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
utils_torch.model.SetMethodForModelClass(__MainClass__)