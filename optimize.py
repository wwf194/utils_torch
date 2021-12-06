import torch
import utils_torch
from utils_torch.attrs import *

from collections import defaultdict

def BuildModuleIfIsLegalType(param, **kw):
    if isinstance(param, str):
        Type = param
    else:
        Type = param.Type
    
    if IsLegalModuleType(Type):
        if Type in ["GradientDescend"]:
            #return GradientDescend(**kw)
            return GradientDescend(**kw)
        else:
            raise Exception(Type)
    else:
        return None

def IsLegalModuleType(Type):
    return Type in ModuleDict

def ParseOptimizeParamEpochBatch(param):
    EnsureAttrs(param, "Nesterov", value=False)
    EnsureAttrs(param, "Dampening", value=0.0)
    EnsureAttrs(param, "Momentum", value=0.0)

class GradientDescend(utils_torch.module.AbstractModuleWithParam):
    # def __init__(self, param=None, data=None, **kw):
    #     self.InitModule(self, param, data, ClassPath="utils_torch.optimize.GradientDescend")
    def __init__(self, **kw):
        super().__init__(**kw)
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        cache = self.cache
        cache.IsLoad = IsLoad
        cache.IsInit = not IsLoad
        if cache.IsInit:
            self.cache.LastUpdateInfo = defaultdict(lambda:{})
        return
    def __call__(self, weights, param, ClearGrad=True, 
            WarnNoneGrad=True, LogWeightChangeRatio=True,
            LogGrad=True, Update=True
        ):
        cache = self.cache
        if LogGrad:
            GradLog = {}
        for Name, Weight in weights.items():
            # if Name in ["Recurrent.FiringRate2RecurrentInput.Weight"]:
            #     print("aaa")
            if Weight.grad is None:
                if WarnNoneGrad:
                    utils_torch.AddWarning("%s.grad is None."%Name)
                continue
            WeightChange = Weight.grad.data
            if LogGrad:
                GradLog[Name] = - Weight.grad.data
            if param.WeightDecay != 0:
                #WeightChange.add_(param.WeightDecay, Weight.data)
                WeightChange.add_(Weight.data, alpha=param.WeightDecay,)
            if param.Momentum != 0:
                LastUpdateInfo = cache.LastUpdateInfo[Weight]
                if 'dW' not in LastUpdateInfo:
                    WeightChangeMomentum = LastUpdateInfo['dW'] = torch.clone(WeightChange).detach()
                else:
                    WeightChangeMomentum = LastUpdateInfo['dW']
                    #WeightChangeMomentum.mul_(param.Momentum).add_(1 - param.Dampening, WeightChange)
                    WeightChangeMomentum.mul_(param.Momentum).add_(WeightChange, alpha=1.0 - param.Dampening, )
                if param.Nesterov:
                    WeightChange = WeightChange.add(param.Momentum, alpha=WeightChangeMomentum)
                else:
                    WeightChange = WeightChangeMomentum
            #Weight.data.add_(-param.LearningRate, WeightChange)
            # if param.LimitWeightChangeRatio:
            #     RatioMax = param.WeightChangeRatioMax
            #     1.0 * torch.where(Weight == 0.0)
            # else:
            # if LogWeightChangeRatio:
            #     utils_torch.GetDataLogger().AddLog("%s.ChangeRatio"%Name,
            #         utils_torch.transform.CalculateWeightChangeRatio(Weight, WeightChange),
            #         Type="WeightChangeRatio"
            #     )
            if Update:
                Weight.data.add_(WeightChange, alpha=-param.LearningRate)
            if ClearGrad:
                Weight.grad.detach_()
                Weight.grad.zero_()
        if LogGrad:
            return GradLog
#utils_torch.transform.SetMethodForTransformModule(GradientDescend)
ModuleDict = {
    "GradientDescend": GradientDescend
}
