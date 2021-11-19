from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.attrs import *
from utils_torch.Modules.SingleLayer import SingleLayer

class LinearLayer(SingleLayer):
    def __init__(self, param=None, data=None, **kw):
        super().__init__()
        utils_torch.module.InitForModule(self, param, data, ClassPath="utils_torch.Modules.LinearLayer", **kw)
    def InitFromParam(self, param=None, IsLoad=False):
        super().InitFromParam(IsLoad)
        param = self.param
        data = self.data
        cache = self.cache

        SetAttrs(param, "Type", value="LinearLayer")
        EnsureAttrs(param, "Subtype", default="Wx+b")     

        if param.Subtype in ["Wx"]:
            self.SetWeight()
            self.forward = lambda x:torch.mm(x, self.GetWeight())
        elif param.Subtype in ["Wx+b"]:
            SetAttrs(param, "Bias", value=True)
            SetAttrs(param, "Bias.Size", value=param.Output.Num)
            self.SetWeight()
            self.SetBias()
            self.forward = lambda x:torch.mm(x, self.GetWeight()) + self.GetBias()         
        elif param.Subtype in ["W(x+b)"]:
            SetAttrs(param, "Bias", value=True)
            SetAttrs(param, "Bias.Size", value=param.Input.Num)
            self.SetWeight()
            self.SetBias()
            self.forward = lambda x:torch.mm(x, self.GetWeight() + self.GetBias())               
        else:
            raise Exception("LinearLayer: Invalid Subtype: %s"%param.Subtype)

__MainClass__ = LinearLayer
#utils_torch.module.SetMethodForModuleClass(__MainClass__)