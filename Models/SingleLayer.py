import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

class SingleLayer(nn.Module):
    def __init__(self):
        super(SingleLayer, self).__init__()
    def InitFromParam(self, param=None):
        if param is None:
            param = self.param
            data = self.data
            cache = self.cache
        else:
            self.param = param
            self.data = utils_torch.json.EmptyPyObj()
            self.cache = utils_torch.json.EmptyPyObj()
        
        cache.ParamIndices = []

        EnsureAttrs(param, "ExciInhi", default=False)

        if not HasAttrs(param, "Output.Num") or not HasAttrs(param, "Input.Num"):
            if HasAttrs(param, "Weight.Size"):
                SetAttrs(param, "Input.Num", param.Weight.Size[0])
                SetAttrs(param, "Output.Num", param.Weight.Size[1])
            else:
                raise Exception()
    def CreateBias(self, Size=None):
        param = self.param
        data = self.data
        cache = self.cache
        if GetAttrs(param.Bias.Enable):
            data.Bias = torch.nn.Parameter(torch.zeros(param.Bias.Size))
            cache.ParamIndices.append([data, "Bias", data.Bias])
        else:
            data.Bias = 0.0
    def CreateWeight(self):
        param = self.param
        data = self.data
        cache = self.cache
        EnsureAttrs(param, "Weight.Size", value=[param.Input.Num, param.Output.Num])
        EnsureAttrs(param, "Weight.Init", default=utils_torch.json.PyObj(
            {"Method":"kaiming", "Coefficient":1.0})
        )
        data.Weight = torch.nn.Parameter(utils_torch.model.CreateWeight2D(param.Weight))
        cache.ParamIndices.append([data, "Weight", data.Weight])
        GetWeightFunction = [lambda :data.Weight]

        EnsureAttrs(param.Weight, "IsExciInhi", default=param.ExciInhi)
        EnsureAttrs(param.Weight, "NoSelfConnection", default=False)
        if param.Weight.IsExciInhi:
            self.ExciInhiMask = utils_torch.model.CreateExcitatoryInhibitoryMask(*param.Weight.Size, param.Weight.excitatory.Num, param.Weight.inhibitory.Num)
            GetWeightFunction.append(lambda Weight:Weight * self.ExciInhiMask)
            EnsureAttrs(param.Weight, "ConstraintMethod", value="AbsoluteValue")
            data.WeightConstraintMethod = utils_torch.model.GetConstraintFunction(param.Weight.ConstraintMethod)
            GetWeightFunction.append(data.WeightConstraintMethod)
        if GetAttrs(param.Weight, "NoSelfConnection")==True:
            if param.Weight.Size[0] != param.Weight.Size[1]:
                raise Exception("NoSelfConnection requires Weight to be square matrix.")
            self.SelfConnectionMask = utils_torch.model.CreateSelfConnectionMask(param.Weight.Size[0])            
            GetWeightFunction.append(lambda Weight:Weight * self.SelfConnectionMask)
        self.GetWeight = utils_torch.StackFunction(GetWeightFunction)
        return
    def SetTensorLocation(self, Location):
        cache = self.cache
        cache.TensorLocation = Location
        utils_torch.model.ListParameter(self)
        for ParamIndex in cache.ParamIndices:
            setattr(ParamIndex[0], ParamIndex[1], ParamIndex[2].to(Location))
    def GetTensorLocation(self):
        return self.cache.TensorLocation
__MainClass__ = SingleLayer