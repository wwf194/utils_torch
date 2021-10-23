import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
#from typing import DefaultDict
from collections import defaultdict
import utils_torch
#Operators = utils_torch.PyObj()
OperatorList = []

def BuildModule(param, **kw):
    if param.Type in ["FunctionsOutputs"]:
        return FunctionsOutputs(param)
    else:
        raise Exception(param.Type)

def IsLegalModuleType(Type):
    if Type in OperatorList:
        return True
    else:
        return False

def Add(*Args):
    # Sum = Args[0]
    # for Index in range(1, len(Args)):
    #     Sum += Args[Index]
    # return Sum
    return sum(Args)
OperatorList.append(["Add"])

def FilterFromDict(Dict, Name):
    return Dict[Name]
OperatorList.append(["FilterFromDict"])

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
    def __init__(self, param=None, data=None, **kw):
        utils_torch.model.InitForModel(self, param, data, 
            ClassPath="utils_torch.Modules.Operators.FunctionsOutputs", **kw)
    def InitFromParam(self, IsLoad=False):
        param = self.param
        cache = self.cache
        cache.IsLoad = IsLoad
        cache.IsInit = not IsLoad
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
utils_torch.model.SetMethodForModelClass(FunctionsOutputs)
OperatorList.append("FunctionsOutputs")

def CalculateGradient(loss):
    loss.backward()
    return
# Operators.CalculateGradient = CalculateGradient
OperatorList.append(["CalculateGradient"])



def CreateDataLogger():
    return utils_torch.log.DataLogger()
OperatorList.append("CreateDataLogger")

def PlotDistribution(Activity, Name="UnNamed"):
    activity = utils_torch.ToNpArray(Activity)
    utils_torch.plot.PlotDistribution1D(activity, Name=Name)

def LogStat(data, Name):
    data = ToNpArray(data)
    statistics = utils_torch.math.NpStatistics(data, ReturnType="Dict")
    utils_torch.GetDataLogger().AddLogDict({statistics})

def Tensor2Statistics2File(data, Name, FilePath=None):
    #Name, FilePath = utils_torch.ParseTextFilePathFromName(Name, FilePath)
    if FilePath is None:
        FilePath = utils_torch.GetMainSaveDir() + Name + "-statistics" + ".txt"
        FilePath = utils_torch.RenameIfPathExists(FilePath)
    statistics = utils_torch.math.TorchTensorStat(data)
    utils_torch.Data2TextFile(statistics, FilePath=FilePath)

from utils_torch.utils import Data2TextFile, ToNpArray
OperatorList.append("Data2TextFile")

from utils_torch.plot import CompareDensityCurve
OperatorList.append("CompareDensityCurve")

from utils_torch.train import ClearGrad
OperatorList.append("ClearGrad")

