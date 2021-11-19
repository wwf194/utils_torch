import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.attrs import *

class L2Loss():
    def __init__(self, param=None, data=None, **kw):
        super(L2Loss, self).__init__()
        utils_torch.module.InitForModule(self, param, data, ClassPath="utils_torch.Loss.L2Loss", **kw)
    def __call__(self, Input, *Args):
        return self.forward(Input, *Args)
    def InitFromParam(self, IsLoad=False):
        utils_torch.module.InitFromParamForModule(self, IsLoad)
        param = self.param
        data = self.data
        cache = self.cache

        data.CoefficientHistory = []
        Coefficient = GetAttrs(param.Coefficient)
        
        if Coefficient in ["Adaptive", "Dynamic"]:
            if param.Coefficient.Method in ["Ratio2RefLoss"]:
                cache.Coefficient = 1.0
                self.GetCoefficient = self.GetCoefficientRatio2RefLoss
                if isinstance(GetAttrs(param.Coefficient.Ratio), list):
                    SetAttrs(param, "Coefficient.Ratio.Min", GetAttrs(param.Coefficient.Ratio)[0])
                    SetAttrs(param, "Coefficient.Ratio.Max", GetAttrs(param.Coefficient.Ratio)[1])
                elif isinstance(param.Coefficient.Ratio, utils_torch.PyObj):
                    pass    
                else:
                    raise Exception()
                cache.RatioMin = param.Coefficient.Ratio.Min
                cache.RatioMax = param.Coefficient.Ratio.Max
                cache.RatioMid = (cache.RatioMin + cache.RatioMin) / 2.0
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
        LossRefFloat = LossRef.item()
        if cache.RatioMin * LossRefFloat <= cache.Coefficient * LossFloat <= cache.RatioMax * LossRefFloat:
            return cache.Coefficient
        else:
            # Coeff * Loss = RaioMid * LossRef
            data.CoefficientHistory.append(cache.Coefficient)
            cache.Coefficient = cache.RatioMid * LossRefFloat / LossFloat
            return cache.Coefficient
    def forward(self, Input, *Args):
        param = self.param
        if isinstance(Input, tuple) or isinstance(Input, list):
            Loss = 0.0
            for _Input in Input:
                Loss += torch.mean(_Input ** 2)
        elif isinstance(Input, torch.Tensor):
            Loss = torch.mean(Input)
        else:
            raise Exception()
        Coefficient = self.GetCoefficient(Loss, *Args)
        Loss = Coefficient * Loss
        return Loss
    def GetCoefficient(self):
        return self.cache.Coefficient

__MainClass__ = L2Loss
utils_torch.module.SetMethodForModuleClass(__MainClass__)