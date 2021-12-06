import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
from collections import defaultdict
import utils_torch
#Operators = utils_torch.PyObj()

ModuleList = []

def BuildModuleIfIsLegalType(param, **kw):
    if isinstance(param, str):
        Type = param
    else:
        Type = param.Type

    if IsLegalModuleType(Type):
        return BuildModule(param, **kw)
    else:
        return None

def BuildModule(param, **kw):
    if isinstance(param, str):
        Type = param
    else:
        Type = param.Type
        
    if Type in ["FunctionsOutputs"]:
        return FunctionsOutputs()
    else:
        raise Exception(Type)

def IsLegalModuleType(Type):
    if Type in ModuleList:
        return True
    else:
        return False

def Add(*Args):
    return sum(Args)
ModuleList.append(["Add"])

def FilterFromDict(Dict, Name):
    return Dict[Name]
ModuleList.append(["FilterFromDict"])

def Split(Args):
    if isinstance(Args, list):
        return Args
    elif isinstance(Args, utils_torch.PyObj) and Args.IsListLike():
        return Args
    else:
        raise Exception
# Operators.Split = Split
ModuleList.append(["Split"])

def Merge(*Args):
    return Args
ModuleList.append(["Merge"])

def FunctionsOutputs2List(Functions):
    Outputs = []
    for Function in Functions:
        Output = utils_torch.CallFunction(Function)
        Outputs.append(Output)
    return Outputs
# Operators.FunctionsOutputs2List = FunctionsOutputs2List

from utils_torch.transform import AbstractTransform
class FunctionsOutputs(AbstractTransform):
    # def __init__(self, param=None, data=None, **kw):
    #     self.InitModule(self, param, data, 
    #         ClassPath="utils_torch.transform.operator.FunctionsOutputs", **kw)
    def __init__(self, **kw):
        super().__init__(**kw)
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
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
#utils_torch.transform.SetMethodForTransformModule(FunctionsOutputs)
ModuleList.append("FunctionsOutputs")

def CalculateGradient(loss):
    loss.backward()
    return
# Operators.CalculateGradient = CalculateGradient
ModuleList.append(["CalculateGradient"])



def CreateDataLogger():
    return utils_torch.log.DataLogger()
ModuleList.append("CreateDataLogger")

def PlotDistribution(Activity, Name="UnNamed"):
    activity = utils_torch.ToNpArray(Activity)
    utils_torch.plot.PlotDistribution1D(activity, Name=Name)

def LogStat(data, Name):
    data = utils_torch.ToNpArray(data)
    statistics = utils_torch.math.NpStatistics(data, ReturnType="Dict")
    utils_torch.GetDataLogger().AddLogDict({statistics})

def Tensor2Statistics2File(data, Name, FilePath=None):
    #Name, FilePath = utils_torch.ParseTextFilePathFromName(Name, FilePath)
    if FilePath is None:
        FilePath = utils_torch.GetMainSaveDir() + Name + "-statistics" + ".txt"
        FilePath = utils_torch.RenameIfFileExists(FilePath)
    statistics = utils_torch.math.TorchTensorStat(data)
    utils_torch.Data2TextFile(statistics, FilePath=FilePath)

ModuleList.append("Data2TextFile")

from utils_torch.plot import CompareDensityCurve
ModuleList.append("CompareDensityCurve")

# from utils_torch.train import ClearGrad
# ModuleList.append("ClearGrad")

# from utils_torch.train import Probability2MostProbableIndex
# ModuleList.append("Probability2MostProbableIndex")

# from utils_torch.transform import LogAccuracyForSingleClassPrediction
# ModuleList.append("LogAccuracyForSingleClassPrediction")