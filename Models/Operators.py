import torch

#from typing import DefaultDict
from collections import defaultdict
import utils_torch
#Operators = utils_torch.PyObj()
OperatorList = []

def BuildModule(param, RaiseIfFail=False):
    if param.Type in ["FunctionsOutputs"]:
        return FunctionsOutputs(param)
    else:
        if RaiseIfFail:
            raise Exception(param.Type)
        else:
            return False

def IsLegalType(Type):
    if Type in OperatorList:
        return True
    else:
        return False

def Add(*Args):
    Sum = Args[0]
    for Index in range(1, len(Args)):
        Sum += Args[Index]
    return Sum
# Operators.Add = Add
OperatorList.append(["Add"])

def Split(Args):
    if isinstance(Args, list):
        return Args
    elif isinstance(Args, utils_torch.PyObj) and Args.IsListLike():
        return Args
    else:
        raise Exception
# Operators.Split = Split
OperatorList.append(["Split"])

def Merge(*Args):
    return Args
OperatorList.append(["Merge"])

def FunctionsOutputs2List(Functions):
    Outputs = []
    for Function in Functions:
        Output = utils_torch.CallFunction(Function)
        Outputs.append(Output)
    return Outputs
# Operators.FunctionsOutputs2List = FunctionsOutputs2List

class FunctionsOutputs:
    def __init__(self, param=None):
        if param is not None:
            self.param = param
            self.cache = utils_torch.EmptyPyObj()
    def InitFromParam(self):
        param = self.param
        cache = self.cache
        utils_torch.ParseFunctionParamsStatic(param.Functions)
        cache.Functions = utils_torch.parse.ParsePyObjDynamic(
            param.Functions,
            ObjCurrent=param.Functions.cache.__ResolveRef__
            # to be implemented: ObjRoot = ?
        )
        return
    def __call__(self):
        return self.forward()
    def forward(self):
        return FunctionsOutputs2List(self.cache.Functions)
OperatorList.append("FunctionsOutputs")

def CalculateGradient(loss):
    loss.backward()
    return
# Operators.CalculateGradient = CalculateGradient
OperatorList.append(["CalculateGradient"])

def Log(data, Name=None):
    if isinstance(data, torch.Tensor):
        statistics = utils_torch.math.TorchTensorStatistics(data)
    else:
        raise Exception()
    return
OperatorList.append("Log")

class GradientDescend:
    def __init__(self, param=None):
        self.cache = utils_torch.EmptyPyObj()
        self.cache.LastUpdateInfo = defaultdict(lambda:{})
    def InitFromParam(self):
        return
    def __call__(self, weights, param, ClearGrad=True, WarnNoneGrad=True):
        cache = self.cache
        for Name, Weight in weights.items():
            if Weight.grad is None:
                if WarnNoneGrad:
                    utils_torch.AddWarning("%s.grad is None."%Name)
                continue
            WeightChange = Weight.grad.data
            if param.WeightDecay != 0:
                WeightChange.add_(param.WeightDecay, Weight.data)
            if param.Momentum != 0:
                LastUpdateInfo = cache.LastUpdateInfo[Weight]
                if 'dW' not in LastUpdateInfo:
                    WeightChangeMomentum = LastUpdateInfo['dW'] = torch.clone(WeightChange).detach()
                else:
                    WeightChangeMomentum = LastUpdateInfo['dW']
                    WeightChangeMomentum.mul_(param.Momentum).add_(1 - param.Dampening, WeightChange)
                if param.Nesterov:
                    WeightChange = WeightChange.add(param.Momentum, WeightChangeMomentum)
                else:
                    WeightChange = WeightChangeMomentum
            Weight.data.add_(-param.LearningRate, WeightChange)
            if ClearGrad:
                Weight.grad.detach_()
                Weight.grad.zero_()        
        return
OperatorList.append("GradientDescend")