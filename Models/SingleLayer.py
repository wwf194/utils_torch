import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.model import *
from utils_torch.utils import *
from utils_torch.utils import HasAttrs, EnsureAttrs, MatchAttrs, StackFunction, SetAttrs
from utils_torch.model import GetNonLinearFunction, GetConstraintFunction, CreateSelfConnectionMask, CreateExcitatoryInhibitoryMask, Create2DWeight

def InitFromParam(param):
    model = SingleLayer()
    model.InitFromParam(param)
    return model

class SingleLayer(nn.Module):
    def __init__(self, param=None):
        super(SingleLayer, self).__init__()
        if param is not None:
            self.param = param
    def InitFromParam(self, param):
        super(SingleLayer, self).__init__()
        SetAttrs(param, "Type", value="SingleLayer")
        EnsureAttrs(param, "Subtype", default="f(Wx+b)")
        self.param = param
        EnsureAttrs(param, "Subtype", default="f(Wx+b)")

        EnsureAttrs(param, "Weight", default=utils_torch.PyObj(
            {"Initialize":{"Method":"kaiming", "Coefficient":1.0}}))

        if not HasAttrs(param.Weight, "Size"):
            SetAttrs(param.Weight, "Size", value=[param.Input.Num, param.Output.Num])
        if param.Subtype in ["f(Wx+b)"]:
            self.CreateWeight()
            self.CreateBias()
            self.NonLinear = GetNonLinearFunction(param.NonLinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.GetWeight()) + self.Bias)
        elif param.Subtype in ["f(Wx)+b"]:
            self.CreateWeight()
            self.CreateBias()
            self.NonLinear = GetNonLinearFunction(param.NonLinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.GetWeight())) + self.Bias
        elif param.Subtype in ["Wx"]:
            self.CreateWeight()
            self.forward = lambda x:torch.mm(x, self.GetWeight())
        elif param.Subtype in ["Wx+b"]:
            self.CreateWeight()
            self.CreateBias()
            self.forward = lambda x:torch.mm(x, self.GetWeight()) + self.Bias         
        else:
            raise Exception("SingleLayer: Invalid Subtype: %s"%param.Subtype)
    def CreateBias(self, Size=None):
        param = self.param
        EnsureAttrs(param, "Bias", default=False)
        if Size is None:
            Size = param.Weight.Size[1]
        if MatchAttrs(param.Bias, value=False):
            self.Bias = 0.0
        elif MatchAttrs(param.Bias, value=True):
            self.Bias = torch.nn.Parameter(torch.zeros(Size))
        else:
            # to be implemented 
            raise Exception()
    def CreateWeight(self):
        param = self.param
        sig = HasAttrs(param.Weight, "Size")
        if not HasAttrs(param.Weight, "Size"):
            SetAttrs(param.Weight, "Size", value=[param.Input.Num, param.Output.Num])
        self.Weight = torch.nn.Parameter(Create2DWeight(param.Weight))
        GetWeight_function = [lambda :self.Weight]
        if MatchAttrs(param.Weight, "isExciInhi", value=True):
            self.ExciInhiMask = CreateExcitatoryInhibitoryMask(*param.Weight.Size, param.Weight.excitatory.Num, param.Weight.inhibitory.Num)
            GetWeight_function.append(lambda Weight:Weight * self.ExciInhiMask)
            EnsureAttrs(param.Weight, "ConstraintMethod", value="AbsoluteValue")
            self.WeightConstraintMethod = GetConstraintFunction(param.Weight.ConstraintMethod)
            GetWeight_function.append(self.WeightConstraintMethod)
        if MatchAttrs(param.Weight, "NoSelfConnection", value=True):
            if param.Weight.Size[0] != param.Weight.Size[1]:
                raise Exception("NoSelfConnection requires Weight to be square matrix.")
            self.SelfConnectionMask = CreateSelfConnectionMask(param.Weight.Size[0])            
            GetWeight_function.append(lambda Weight:Weight * self.SelfConnectionMask)
        self.GetWeight = StackFunction(GetWeight_function)
    
__MainClass__ = SingleLayer