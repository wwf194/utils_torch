import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.attrs import *

class L2Loss(nn.Module):
    def __init__(self, param=None):
        super(L2Loss, self).__init__()
        if param is not None:
            self.param = param
            self.data = utils_torch.EmptyPyObj()
    def InitFromParam(self):
        param = self.param
        data = self.data
        cache = self.cache

        data.CoefficientList = []
        Coefficient = GetAttrs(param.Coefficient)
        if Coefficient in ["Adaptive"]:
            if param.Coefficient.Method in ["Ratio2RefLoss"]:
                self.GetCoefficient = self.GetCoefficientDefault
                if isinstance(param.Coefficient.Ratio, list):
                    SetAttrs(param, "Coefficient.Ratio.Min", GetAttrs(param.Coefficient.Ratio)[0])
                    SetAttrs(param, "Coefficient.Ratio.Max", GetAttrs(param.Coefficient.Ratio)[1])
                elif isinstance(param.Coefficient.Ratio, utils_torch.PyObj):
                    pass    
                else:
                    raise Exception()
                cache.RaiotMin = param.Coefficient.Ratio.Min
                cache.RatioMax = param.Coefficient.Ratio.Max
                cache.RatioMid = (cache.RatioMin + cache.RatioMid) / 2.0
            else:
                raise Exception()
        elif isinstance(Coefficient, float):
            cache.Coefficient = Coefficient
            self.GetCoefficient = self.GetCoefficientDefault
        else:
            raise Exception(Coefficient)
        return
    
    def GetCoefficientDefault(self, *Args):
        return self.cache.Coefficient
    def GetCoefficientRatio2RefLoss(self, Loss, LossRef):
        data = self.data
        cache = self.cache
        LossFloat = Loss.item()
        LossRefFloat = Loss.item()
        if cache.RatioMin * LossRefFloat <= cache.Coefficient * LossFloat <= cache.RatioMax * LossRefFloat:
            return cache.Coefficient
        else:
            # Coeff * Loss = RaioMid * LossRef
            data.CoefficientHistory.append(cache.Coefficient)
            cache.Coefficient = cache.Mid * LossRefFloat / LossFloat
            return cache.Coefficient
    def forward(self, InputList, *Args):
        param = self.param
        Loss = 0.0
        for Input in InputList:
            Loss += torch.sum(Input ** 2)
        Coefficient = self.GetCoefficient(Loss, *Args)
        Loss = Coefficient * Loss
        return Loss
    def GetCoefficient(self):
        return self.cache.Coefficient

__MainClass__ = L2Loss