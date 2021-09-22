import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.model import *
from utils_torch.utils import *
from utils_torch.utils import HasAttrs, EnsureAttrs, MatchAttrs, StackFunction, SetAttrs
from utils_torch.model import GetNonLinearFunction, GetConstraintFunction, CreateSelfConnectionMask, CreateExcitatoryInhibitoryMask, CreateWeight2D

def InitFromParam(param):
    model = SingleLayer()
    model.InitFromParam(param)
    return model

class SingleLayer(nn.Module):
    def __init__(self, param=None):
        super(SingleLayer, self).__init__()
        if param is not None:
            self.param = param
            self.data = utils_torch.json.EmptyPyObj()
            self.cache = utils_torch.json.EmptyPyObj()
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
        SetAttrs(param, "Type", value="SingleLayer")
        EnsureAttrs(param, "Subtype", default="f(Wx+b)")
        self.param = param
        EnsureAttrs(param, "Subtype", default="f(Wx+b)")

        EnsureAttrs(param, "Weight", default=utils_torch.json.PyObj(
            {"Init":{"Method":"kaiming", "Coefficient":1.0}}))

        if not HasAttrs(param.Weight, "Size"):
            SetAttrs(param.Weight, "Size", value=[param.Input.Num, param.Output.Num])
        if param.Subtype in ["f(Wx+b)"]:
            self.CreateWeight()
            self.CreateBias()
            self.NonLinear = GetNonLinearFunction(param.NonLinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.GetWeight()) + data.Bias)
        elif param.Subtype in ["f(Wx)+b"]:
            self.CreateWeight()
            self.CreateBias()
            self.NonLinear = GetNonLinearFunction(param.NonLinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.GetWeight())) + data.Bias
        elif param.Subtype in ["Wx"]:
            self.CreateWeight()
            self.forward = lambda x:torch.mm(x, self.GetWeight())
        elif param.Subtype in ["Wx+b"]:
            self.CreateWeight()
            self.CreateBias()
            self.forward = lambda x:torch.mm(x, self.GetWeight()) + data.Bias         
        else:
            raise Exception("SingleLayer: Invalid Subtype: %s"%param.Subtype)
    def CreateBias(self, Size=None):
        param = self.param
        data = self.data
        cache = self.cache
        EnsureAttrs(param, "Bias", default=False)
        if Size is None:
            Size = param.Weight.Size[1]
        if MatchAttrs(param.Bias, value=False):
            data.Bias = 0.0
        elif MatchAttrs(param.Bias, value=True):
            data.Bias = torch.nn.Parameter(torch.zeros(Size))
            cache.ParamIndices.append([data, "Bias", data.Bias])
        else:
            # to be implemented 
            raise Exception()
    def CreateWeight(self):
        param = self.param
        data = self.data
        cache = self.cache
        sig = HasAttrs(param.Weight, "Size")
        if not HasAttrs(param.Weight, "Size"):
            SetAttrs(param.Weight, "Size", value=[param.Input.Num, param.Output.Num])
        data.Weight = torch.nn.Parameter(CreateWeight2D(param.Weight))
        cache.ParamIndices.append([data, "Weight", data.Weight])

        GetWeightFunction = [lambda :data.Weight]
        if MatchAttrs(param.Weight, "IsExciInhi", value=True):
            self.ExciInhiMask = CreateExcitatoryInhibitoryMask(*param.Weight.Size, param.Weight.excitatory.Num, param.Weight.inhibitory.Num)
            GetWeightFunction.append(lambda Weight:Weight * self.ExciInhiMask)
            EnsureAttrs(param.Weight, "ConstraintMethod", value="AbsoluteValue")
            data.WeightConstraintMethod = GetConstraintFunction(param.Weight.ConstraintMethod)
            GetWeightFunction.append(data.WeightConstraintMethod)
        if MatchAttrs(param.Weight, "NoSelfConnection", value=True):
            if param.Weight.Size[0] != param.Weight.Size[1]:
                raise Exception("NoSelfConnection requires Weight to be square matrix.")
            self.SelfConnectionMask = CreateSelfConnectionMask(param.Weight.Size[0])            
            GetWeightFunction.append(lambda Weight:Weight * self.SelfConnectionMask)
        self.GetWeight = StackFunction(GetWeightFunction)
        return
    def SetTensorLocation(self, Location):
        cache = self.cache
        cache.TensorLocation = Location
        utils_torch.model.ListParameter(self)
        for ParamIndex in cache.ParamIndices:
            setattr(ParamIndex[0], ParamIndex[1], ParamIndex[2].to(Location))
    def GetTensorLocation(self, Location):
        return self.cache.TensorLocation
__MainClass__ = SingleLayer