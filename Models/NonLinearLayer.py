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
    def __init__(self, param=None):
        super().__init__()
        if param is not None:
            self.param = param
            self.data = utils_torch.EmptyPyObj()
            self.cache = utils_torch.EmptyPyObj()
    def InitFromParam(self, param=None):
        super().InitFromParam(param)
        param = self.param        
        data = self.data

        SetAttrs(param, "Type", value="NonLinearLayer")
        EnsureAttrs(param, "Subtype", default="f(Wx+b)")     

        if param.Subtype in ["f(Wx+b)"]:
            SetAttrs(param, "Bias.Enable", True)
            SetAttrs(param, "Bias.Size", param.Output.Num)
            self.CreateWeight()
            self.CreateBias()
            self.NonLinear = utils_torch.model.GetNonLinearMethod(param.NonLinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.GetWeight()) + data.Bias)
        elif param.Subtype in ["f(Wx)+b"]:
            SetAttrs(param, "Bias.Enable", True)
            SetAttrs(param, "Bias.Size", param.Output.Num)
            self.CreateWeight()
            self.CreateBias()
            self.NonLinear = utils_torch.model.GetNonLinearMethod(param.NonLinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.GetWeight())) + data.Bias
        elif param.Subtype in ["Wx"]:
            SetAttrs(param, "Bias.Enable", False)
            self.CreateWeight()
            self.forward = lambda x:torch.mm(x, self.GetWeight())
        elif param.Subtype in ["f(W(x+b))"]:
            SetAttrs(param, "Bias.Enable", True)
            SetAttrs(param, "Bias.Size", param.Input.Num)
            self.CreateWeight()
            self.CreateBias()
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.GetWeight()) + data.Bias)       
        else:
            if param.Subtype in ["Wx", "Wx+b"]:
                raise Exception("NonLinearLayer: Invalid Subtype. Try using LinearLayer.: %s"%param.Subtype)
            else:
                raise Exception("NonLinearLayer: Invalid Subtype: %s"%param.Subtype)
__MainClass__ = NonLinearLayer
utils_torch.model.SetMethodForModelClass(__MainClass__)