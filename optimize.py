import torch
import utils_torch

from collections import defaultdict

class GradientDescend:
    def __init__(self, param=None, data=None, **kw):
        utils_torch.model.InitForModel(self, param, data, ClassPath="utils_torch.optimize.GradientDescend")
    def InitFromParam(self, IsLoad=False):
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
            #         utils_torch.model.CalculateWeightChangeRatio(Weight, WeightChange),
            #         Type="WeightChangeRatio"
            #     )
            if Update:
                Weight.data.add_(WeightChange, alpha=-param.LearningRate)
            if ClearGrad:
                Weight.grad.detach_()
                Weight.grad.zero_()
        if LogGrad:
            return GradLog


utils_torch.model.SetMethodForModelClass(GradientDescend)