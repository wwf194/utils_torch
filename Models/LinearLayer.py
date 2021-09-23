from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.attrs import *
from utils_torch.model import GetNonLinearMethod, GetConstraintFunction, CreateSelfConnectionMask, CreateExcitatoryInhibitoryMask, CreateWeight2D

from utils_torch.Models.SingleLayer import SingleLayer

class LinearLayer(SingleLayer):
    def __init__(self, param=None):
        super().__init__()
        if param is not None:
            self.param = param
            self.data = utils_torch.json.EmptyPyObj()
            self.cache = utils_torch.json.EmptyPyObj()
    def InitFromParam(self, param=None):
        super().InitFromParam(param)
        param = self.param        
        data = self.data

        SetAttrs(param, "Type", value="LinearLayer")
        EnsureAttrs(param, "Subtype", default="Wx+b")     

        if param.Subtype in ["Wx"]:
            self.CreateWeight()
            self.forward = lambda x:torch.mm(x, self.GetWeight())
        elif param.Subtype in ["Wx+b"]:
            SetAttrs(param, "Bias.Enable", value=True)
            SetAttrs(param, "Bias.Size", value=param.Output.Num)
            self.CreateWeight()
            self.CreateBias()
            self.forward = lambda x:torch.mm(x, self.GetWeight()) + data.Bias         
        elif param.Subtype in ["W(x+b)"]:
            SetAttrs(param, "Bias.Enable", value=True)
            SetAttrs(param, "Bias.Size", value=param.Input.Num)
            self.CreateWeight()
            self.CreateBias()
            self.forward = lambda x:torch.mm(x, self.GetWeight() + + data.Bias)               
        else:
            raise Exception("LinearLayer: Invalid Subtype: %s"%param.Subtype)

__MainClass__ = LinearLayer