import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

def InitFromParam(param):
    model = NonLinearLayer()
    model.InitFromParam(param)
    return model

from utils_torch.Models.SingleLayer import SingleLayer

class NonLinearLayer(SingleLayer):
    def __init__(self, param=None, data=None, **kw):
        super().__init__()
        utils_torch.model.InitForModel(self, param, data, ClassPath="utils_torch.Models.NonLinearLayer", **kw)
    def InitFromParam(self, param=None, IsLoad=False):
        super().InitFromParam(param, IsLoad)
        param = self.param        
        data = self.data
        cache = self.cache
        cache.IsLoad = IsLoad
        cache.IsInit = not cache.IsLoad

        SetAttrs(param, "Type", value="NonLinearLayer")
        EnsureAttrs(param, "Subtype", default="f(Wx+b)")     

        if param.Subtype in ["f(Wx+b)"]:
            if cache.IsInit:
                SetAttrs(param, "Bias", True)
                SetAttrs(param, "Bias.Size", param.Output.Num)
            self.SetWeight()
            self.SetBias()
            self.NonLinear = utils_torch.model.GetNonLinearMethod(param.NonLinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.GetWeight()) + self.GetBias())
        elif param.Subtype in ["f(Wx)+b"]:
            if cache.IsInit:
                SetAttrs(param, "Bias", True)
                SetAttrs(param, "Bias.Size", param.Output.Num)
            self.SetWeight()
            self.SetBias()
            self.NonLinear = utils_torch.model.GetNonLinearMethod(param.NonLinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.GetWeight())) + data.Bias
        elif param.Subtype in ["Wx"]:
            if cache.IsInit:
                SetAttrs(param, "Bias", False)
            self.SetWeight()
            self.forward = lambda x:torch.mm(x, self.GetWeight())
        elif param.Subtype in ["f(W(x+b))"]:
            if cache.IsInit:
                SetAttrs(param, "Bias", True)
                SetAttrs(param, "Bias.Size", param.Input.Num)
            self.SetWeight()
            self.SetBias()
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.GetWeight()) + data.Bias)       
        else:
            if param.Subtype in ["Wx", "Wx+b"]:
                raise Exception("NonLinearLayer: Invalid Subtype. Try using LinearLayer.: %s"%param.Subtype)
            else:
                raise Exception("NonLinearLayer: Invalid Subtype: %s"%param.Subtype)
__MainClass__ = NonLinearLayer
#utils_torch.model.SetMethodForModelClass(__MainClass__)