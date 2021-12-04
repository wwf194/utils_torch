import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import utils_torch

from utils_torch.module.AbstractModules import AbstractModule, AbstractModuleWithTensor, AbstractModuleForEpochBatchTrain

ModuleList = [
    "LinearLayer", "NonLinearLayer",
    "NoiseGenerator",
    "RecurrentLIFLayer",
    "LambdaLayer", "Lambda",
    "MLP",
    "SerialSender", "SerialReceiver", "SignalHolder",
    "NonLinear", # NonLinear Function
    "Bias",
]
def IsLegalModuleType(Type):
    return Type in ModuleList

def BuildModuleIfIsLegalType(param, **kw):
    if not IsLegalModuleType(param.Type):
        return None
    if param.Type in ["LinearLayer"]:
        return LinearLayer(param, **kw)
    elif param.Type in ["NonLinearLayer"]:
        return NonLinearLayer(param, **kw)
    elif param.Type in ["MLP", "MultiLayerPerceptron", "mlp"]:
        return MLP(param, **kw)
    elif param.Type in ["SerialReceiver"]:
        return SerialReceiver(param, **kw)
    elif param.Type in ["SerialSender"]:
        return SerialSender(param, **kw)
    elif param.Type in ["SignalHolder"]:
        return SignalHolder(param, **kw)
    elif param.Type in ["Lambda", "LambdaLayer"]:
        return LambdaLayer(param, **kw)
    elif param.Type in ["RecurrentLIFLayer"]:
        return RecurrentLIFLayer(param, **kw)
    elif param.Type in ["NoiseGenerator"]:
        return NoiseGenerator(param, **kw)
    elif param.Type in ["Bias"]:
        return Bias(param, **kw)
    elif param.Type in ["NonLinear"]:
        return GetNonLinearMethod(param, **kw)
    elif param.Type in ["GradientDescend"]:
        return utils_torch.optimize.GradientDescend(param, **kw)
    elif param.Type in ["CheckPointForEpochBatchTrain"]:
        return utils_torch.train.CheckPointForEpochBatchTrain(param, **kw)
    elif param.Type in ["Internal"]:
        utils_torch.AddWarning("utils_torch.transform.BuildModule does not build Module of type Internal.")
        raise Exception()
    elif param.Type in ["External"]:
        utils_torch.AddWarning("utils_torch.transform.BuildModule does not build Module of type External.")
        raise Exception()
    else:
        raise Exception("BuildModule: No such module: %s"%param.Type)


def GetNonLinearMethod(param, **kw):
    param = ParseNonLinearMethod(param)
    if param.Type in ["NonLinear"]:
        if hasattr(param, "Subtype"):
            Type = param.Subtype
    else:
        Type = param.Type

    if Type in ["relu", "ReLU"]:
        if param.Coefficient==1.0:
            return F.relu
        else:
            return lambda x:param.Coefficient * F.relu(x)
    elif Type in ["tanh", "Tanh"]:
        if param.Coefficient==1.0:
            return F.tanh
        else:
            return lambda x:param.Coefficient * F.tanh(x)       
    elif Type in ["sigmoid", "Sigmoid"]:
        if param.Coefficient==1.0:
            return F.tanh
        else:
            return lambda x:param.Coefficient * F.tanh(x)         
    else:
        raise Exception("GetNonLinearMethod: Invalid nonlinear function Type: %s"%param.Type)
GetActivationFunction = GetNonLinearMethod

def ParseNonLinearMethod(param):
    if isinstance(param, str):
        param = utils_torch.PyObj({
            "Type": param,
            "Coefficient": 1.0
        })
    elif isinstance(param, list):
        if len(param)==2:
            param = utils_torch.PyObj({
                "Type": param[0],
                "Coefficient": param[1]
            })
        else:
            # to be implemented
            pass
    elif isinstance(param, utils_torch.PyObj):
        if not hasattr(param, "Coefficient"):
            param.Coefficient = 1.0
    else:
        raise Exception("ParseNonLinearMethod: invalid param Type: %s"%type(param))
    return param

ModuleList = set(ModuleList)


import re
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import utils_torch
from utils_torch.json import *
from utils_torch.attrs import *
from utils_torch.LRSchedulers import LinearLR

def BuildModule(param, **kw):
    if hasattr(param, "ClassPath"):
        Class = utils_torch.parse.ParseClass(param.ClassPath)
        Module = Class(param, **kw)
        return Module
    elif hasattr(param, "Type"):
        return BuildModuleFromType(param, **kw)
    else:
        raise Exception()


def CalculateWeightChangeRatio(Weight, WeightChange):
    Weight = utils_torch.ToNpArray(Weight)
    WeightChange = utils_torch.ToNpArray(WeightChange)

    WeightAbsSum = np.sum(np.abs(Weight))
    if WeightAbsSum > 0.0:
        WeightChangeRatio = np.sum(np.abs(WeightChange)) / WeightAbsSum
    else:
        WeightChangeRatio = float("inf")
    return WeightChangeRatio

def ListParameter(model):
    for name, param in model.named_parameters():
        utils_torch.AddLog("%s: Shape: %s"%(name, param.size()))

def CreateSelfConnectionMask(Size):
    return np.ones((Size, Size), dtype=np.float32) - np.eye(Size, dtype=np.float32)

def CreateExcitatoryInhibitoryMask(InputNum, OutputNum, ExcitatoryNum, InhibitoryNum=None):
    # Assumed weight matrix shape: [InputNum, OutputNum]
    if InhibitoryNum is None:
        InhibitoryNum = InputNum - ExcitatoryNum
    else:
        if InputNum != ExcitatoryNum + InhibitoryNum:
            raise Exception("GetExcitatoryInhibitoryMask: InputNum == ExcitatoryNum + InhibitoryNum must be satisfied.")

    ExcitatoryMask = np.full((ExcitatoryNum, OutputNum), fill_value=1.0, dtype=np.float32)
    InhibitoryMask = np.full((InhibitoryNum, OutputNum), fill_value=-1.0, dtype=np.float32)
    ExcitatoryInhibitoryMask = np.concatenate([ExcitatoryMask, InhibitoryMask], axis=0)
    return ExcitatoryInhibitoryMask

def ParseExciInhiNum(param):
    if not HasAttrs(param, "Excitatory.Num"):
        EnsureAttrs(param, "Excitatory.Ratio", default=0.8)
        if hasattr(param, "Size"):
            SetAttrs(param, "Excitatory.Num", value=round(param.Excitatory.Ratio * param.Size[0]))
            SetAttrs(param, "Inhibitory.Num", value=param.Size[0] - param.Excitatory.Num)
        elif hasattr(param, "Num"):
            SetAttrs(param, "Excitatory.Num", value=round(param.Excitatory.Ratio * param.Num))
            SetAttrs(param, "Inhibitory.Num", value=param.Num - param.Excitatory.Num)     
        else:
            raise Exception()

    return
def CreateMask(N_num, OutputNum, device=None):
    if device is None:
        device = torch.device('cpu')
    mask = torch.ones((N_num, OutputNum), device=device, requires_grad=False)
    return mask

def GetConstraintFunction(Method):
    if Method in ["AbsoluteValue", "abs"]:
        return lambda x:torch.abs(x)
    elif Method in ["Square", "square"]:
        return lambda x:x ** 2
    elif Method in ["CheckAfterUpdate", "force"]:
        return lambda x:x
    else:
        raise Exception("GetConstraintFunction: Invalid consraint Method: %s"%Method)


def CreateWeight2D(param, DataType=torch.float32):
    Init = param.Init
    if Init.Method in ["Kaiming", "KaimingUniform", "KaimingNormal"]:
        if Init.Method in ["KaimingNormal"]: #U~(-bound, bound), bound = sqrt(6/(1+a^2)*FanIn)
            SetAttrs(Init, "Distribution", value="Normal")
        elif Init.Method in ["KaimingUniform"]:
            SetAttrs(Init, "Distribution", value="Uniform")
        else:
            EnsureAttrs(Init, "Distribution", default="Uniform")
        EnsureAttrs(Init, "Mode", default="In")
        EnsureAttrs(Init, "Coefficient", default=1.0)
        if Init.Mode in ["In"]:
            if Init.Distribution in ["Uniform"]:
                Init.Range = [
                    - Init.Coefficient * (6.0 / param.Size[0]) ** 0.5,
                    Init.Coefficient * (6.0 / param.Size[0]) ** 0.5
                ]
                weight = np.random.uniform(*Init.Range, tuple(param.Size))
            elif Init.Distribution in ["Uniform+"]:
                Init.Range = [
                    0.0,
                    2.0 * Init.Coefficient * 6.0 ** 0.5 / param.Size[0] ** 0.5
                ]
                weight = np.random.uniform(*Init.Range, tuple(param.Size))
            elif Init.Distribution in ["Normal"]:
                # std = sqrt(2 / (1 + a^2) * FanIn)
                Mean = 0.0
                Std = Init.Coefficient * (2.0 / param.Size[0]) ** 0.5
                weight = np.random.normal(Mean, Std, tuple(param.Size))
            else:
                # to be implemented
                raise Exception()
        else:
            raise Exception()
            # to be implemented
    elif Init.Method in ["xaiver", "glorot"]:
        Init.Method = "xaiver"
        raise Exception()
        # to be implemented
    else:
        raise Exception()
        # to be implemented
    return utils_torch.NpArray2Tensor(weight, DataType=DataType, RequiresGrad=True)

def GetLossFunction(LossFunctionDescription, truth_is_label=False, num_class=None):
    if LossFunctionDescription in ['MSE', 'mse']:
        if truth_is_label:
            #print('num_class: %d'%num_class)
            #return lambda x, y:torch.nn.MSELoss(x, scatter_label(y, num_class=num_class))
            return lambda x, y:F.mse_loss(x, scatter_label(y, num_class=num_class))
        else:
            return torch.nn.MSELoss()
    elif LossFunctionDescription in ['CEL', 'cel']:
        return torch.nn.CrossEntropyLoss()
    else:
        raise Exception('Invalid loss function description: %s'%LossFunctionDescription)

Getloss_function = GetLossFunction

def Getact_func_module(act_func_str):
    name = act_func_str
    if act_func_str in ['relu']:
        return nn.ReLU()
    elif act_func_str in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif act_func_str in ['softplus', 'SoftPlus']:
        return nn.Softplus()
    elif act_func_str in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    else:
        raise Exception('Invalid act func str: %s'%act_func_str)
        
def Getact_func_from_str(name='relu', Param=None):
    if Param is None:
        Param = 'default'
    if name in ['none']:
        return lambda x:x
    elif name in ['relu']:
        #print(Param)
        if Param in ['default']:
            return lambda x: F.relu(x)
        else:
            return lambda x: Param * F.relu(x)
    elif name in ['tanh']:
        if Param in ['default']:
            return lambda x:torch.tanh(x)
        else:
            return lambda x:Param * F.tanh(x)
    elif name in ['relu_tanh']:
        if Param in ['default']:
            return lambda x:F.relu(torch.tanh(x))
        else:
            return lambda x:Param * F.relu(torch.tanh(x))
    else:
        raise Exception('Invalid act func name: %s'%name)

def build_optimizer(dict_, Params=None, model=None, load=False):
    Type_ = GetFromDict(dict_, 'Type', default='sgd', write_default=True)
    #func = dict_['func'] #forward ; rec, output, input
    #lr = GetFromDict(dict_, 'lr', default=1.0e-3, write_default=True)
    lr = dict_['lr']
    weight_decay = GetFromDict(dict_, 'weight_decay', default=0.0, write_default=True)
    if Params is not None:
        pass
    elif model is not None:
        if hasattr(model, 'GetParam_to_train'):
            Params = model.GetParam_to_train()
        else:
            Params = model.Parameters()
    else:
        raise Exception('build_optimizer: Both Params and model are None.')
    
    if Type_ in ['adam', 'ADAM']:
        optimizer = optim.Adam(Params, lr=lr, weight_decay=weight_decay)
    elif Type_ in ['rmsprop', 'RMSProp']:
        optimizer = optim.RMSprop(Params, lr=lr, weight_decay=weight_decay)
    elif Type_ in ['sgd', 'SGD']:
        momentum = dict_.setdefault('momentum', 0.9)
        optimizer = optim.SGD(Params, momentum=momentum, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception('build_optimizer: Invalid optimizer Type: %s'%Type_)

    if load:
        optimizer.load_state_dict(dict_['state_dict'])
    else:
        dict_['state_dict'] = optimizer.state_dict()
    return optimizer


# search for directory or file of most recently saved models(model with biggest epoch index)
def GetLastestSubSaveDir(SaveDir, Prefix=None):
    if is_dir:
        max_epoch = None
        pattern = model_prefix+'(\d+)'
        dirs = os.listdir(base_dir)
        for dir_name in dirs:
            result = re.search(r''+pattern, dir_name)
            if result is not None:
                try:
                    epoch_num = int(result.group(1))
                except Exception:
                    print('error in matching model name.')
                    continue
                if max_epoch is None:
                    max_epoch = epoch_num
                else:
                    if max_epoch < epoch_num:
                        max_epoch = epoch_num
    if max_epoch is not None:
        return base_dir + model_prefix + str(max_epoch) + '/'
    else:
        return 'error'

def cal_acc_from_label(output, label):
    # output: [batch_Size, num_class]; label: [batch_Size], label[i] is the index of correct category of i_th batch.
    correct_num = (torch.max(output, dim=1)[1]==label).sum().item()
    sample_num = label.Size(0)
    #return {'correct_num':correct_num, 'data_num':label_num} 
    return correct_num, sample_num

def scatter_label(label, num_class=None, device=None): # label: must be torch.LongTensor, shape: [batch_Size], label[i] is the index of correct category of i_th batch.
    #print('aaa')
    if num_class is None:
        #print(torch.max(label).__class__)
        num_class = torch.max(label).item() + 1
    scattered_label = torch.zeros((label.Size(0), num_class), device=device).to(label.device).scatter_(1, torch.unsqueeze(label, 1), 1)
    #return scattered_label.long() # [batch_Size, num_class]
    #print(label.Type())
    #print(scattered_label.Type())
    return scattered_label # [batch_Size, num_class]

def print_model_Param(model):
    for name, Param in model.named_Parameters():
        print(Param)
        print('This is my %s. Size:%s is_leaf:%s device:%s requires_grad:%s'%
            (name, list(Param.Size()), Param.is_leaf, Param.device, Param.requires_grad))

def TorchTensorInfo(tensor, name='', verbose=True, complete=True):
    print(tensor.device)
    report = '%s...\033[0;31;40mVALUE\033[0m\n'%str(tensor)
    if complete:
        report += '%s...\033[0;31;40mGRADIENT\033[0m\n'%str(tensor.grad)
    report += 'Tensor \033[0;32;40m%s\033[0m: Size:%s is_leaf:%s device:%s Type:%s requires_grad:%s'%\
        (name, list(tensor.Size()), tensor.is_leaf, tensor.device, tensor.Type(), tensor.requires_grad)
    if verbose:
        print(report)
    return report

def PrintStateDict(optimizer):
    dict_ = optimizer.state_dict()
    for key, value in dict_.items():
        print('%s: %s'%(key, value))


def SetTrainWeightForModule(self):
    ClearTrainWeightForModule(self)
    cache = self.cache
    cache.TrainWeight = {}
    if hasattr(cache, "Modules"):
        for ModuleName, Module in utils_torch.ListAttrsAndValues(cache.Modules):
            if hasattr(Module, "SetTrainWeight"):
                TrainWeight = Module.SetTrainWeight()
                for name, weight in TrainWeight.items():
                    cache.TrainWeight[ModuleName + "." + name] = weight
            else:
                if isinstance(Module, nn.Module):
                    utils_torch.AddWarning("Module %s is instance of nn.Module, but has not implemented GetTrainWeight method."%Module)
        return cache.TrainWeight
    else:
        return {}

def GetPlotWeightForModule(self):
    cache = self.cache
    if not hasattr(cache, "PlotWeight"):
        self.SetPlotWeight()
    weights = {}
    for name, method in cache.PlotWeight.items():
        weights[name] = method()
    return weights

def SetPlotWeightForModule(self):
    ClearPlotWeightForModule(self)
    cache = self.cache
    cache.PlotWeight = {}
    if hasattr(cache, "Modules"):
        for ModuleName, Module in utils_torch.ListAttrsAndValues(cache.Modules):
            if hasattr(Module, "GetPlotWeight"):
                PlotWeightMethod = Module.SetPlotWeight()
                for name, method in PlotWeightMethod.items():
                    cache.PlotWeight[ModuleName + "." + name] = method
            else:
                if isinstance(Module, nn.Module):
                    utils_torch.AddWarning("Module %s is instance of nn.Module, but has not implemented GetTrainWeight method."%Module)
        return cache.PlotWeight
    else:
        return {}

def ClearPlotWeightForModule(self):
    cache = self.cache
    if hasattr(cache, "PlotWeight"):
        delattr(cache, "PlotWeight")

def ClearTrainWeightForModule(self):
    cache = self.cache
    if hasattr(cache, "TrainWeight"):
        delattr(cache, "TrainWeight")

def SetlogForModule(self, log):
    cache = self.cache
    cache.log = log
    if hasattr(cache, "Modules"):   
        for Name, Module in ListAttrsAndValues(self.cache.Modules):
            if hasattr(Module, "Setlog"):
                Module.Setlog(utils_torch.log.Datalog().SetParent(log, prefix=Name + "."))

def SetFullNameForModule(self, FullName):
    cache = self.cache
    param = self.param
    if FullName not in [""]:
        param.FullName = FullName
    if hasattr(cache, "Modules"):   
        for Name, Module in ListAttrsAndValues(cache.Modules):
            if hasattr(Module, "SetFullName"):
                if FullName in [""]:
                    Module.SetFullName(Name)
                else:
                    Module.SetFullName(FullName + "." + Name)

def GetFullNameForModule(self):
    if hasattr(self, "param"):
        if hasattr(self.param, "FullName"):
            return self.param.FullName
    return None

def GetlogForModule(self):
    cache = self.cache
    if hasattr(cache, "log"):
        return cache.log
    else:
        return None

def InitFromParamForModule(self, IsLoad):
    cache = self.cache
    cache.IsLoad = IsLoad
    cache.IsInit = not IsLoad
    cache.__object__ = self

    self.Modules = cache.Modules
    self.Dynamics = cache.Dynamics

    param = self.param
    utils_torch.parse.ParsePyObjStatic(param, ObjCurrent=param)

def InitFromParamForNonModel(self, IsLoad):
    cache = self.cache
    cache.IsLoad = IsLoad
    cache.IsInit = not IsLoad
    cache.__object__ = self

    self.Modules = cache.Modules
    self.Dynamics = cache.Dynamics

    utils_torch.parse.ParsePyObjStatic(self.param)

def DoTasksForModule(Tasks, **kw):
    kw["DoNotChangeObjCurrent"] = True
    utils_torch.DoTasks(Tasks, **kw)

def InitForNonModel(self, param=None, data=None, ClassPath=None, **kw):
    InitForModule(self, param, data, ClassPath, HasTensor=False, **kw)
    return

def InitForModule(self, param=None, data=None, ClassPath=None, **kw):
    LoadDir = kw.get("LoadDir")
    FullName = kw.setdefault("FullName", "Unnamed")
    HasTensor = kw.setdefault("HasTensor", True)

    if param is None:
        param = utils_torch.EmptyPyObj()
    
    EnsureAttrs(param, "FullName", default=FullName)

    param.cache.__object__ = self
    if hasattr(param, "Modules"):
        param.Modules.SetResolveBase()
    if hasattr(param, "Dynamics"):
        param.Dynamics.SetResolveBase()

    if data is None:
        if LoadDir is not None:
            DataPath = LoadDir + param.FullName + ".data"
            if utils_torch.FileExists(DataPath):
                data = utils_torch.json.DataFile2PyObj(DataPath)
            else:
                data = utils_torch.EmptyPyObj()
        else:
            data = utils_torch.EmptyPyObj()

    cache = utils_torch.EmptyPyObj()
    if LoadDir is not None:
        cache.LoadDir = LoadDir
    else:
        cache.LoadDir = None
    if ClassPath is not None:
        param.ClassPath = ClassPath
    
    cache.Modules = utils_torch.EmptyPyObj()
    cache.Dynamics = utils_torch.EmptyPyObj()

    if HasTensor:
        cache.Tensors = []

    self.param = param
    self.data = data
    self.cache = cache
    self.Modules = cache.Modules
    self.Dynamics = cache.Dynamics

def LogDictForModule(self, Dict, Name, Type=None, log="Data"):
    log = utils_torch.ParseLog(log)
    param = self.param
    if hasattr(param, "FullName"):
        Name = param.FullName + "." + Name
    log.AddLogDict(Name, Dict, Type)

def LogStatForModule(self, data, Name, Type="Stat", log="Data"):
    log = utils_torch.ParseLog(log)
    param = self.param
    if hasattr(param, "FullName"):
        Name = param.FullName + "." + Name
    data = utils_torch.ToNpArray(data)
    stat = utils_torch.math.NpStatistics(data, ReturnType="Dict")
    if not Name.endswith("Stat"):
        Name += "-Stat"
    log.AddLogDict(Name, stat, Type)

def LogActivityStatForModule(self, data, Name, Type="Activity-Stat", log="Data"):
    LogStatForModule(self, data, Name, Type=Type, log=log)

def LogWeightStatForModule(self, weights, Name, Type="Weight-Stat", log="Data"):
    log = utils_torch.ParseLog(log)
    param = self.param
    for Name, Weight in weights.items():
        WeightStat = utils_torch.math.TorchTensorStat(Weight, ReturnType="Dict")
        log.AddLogDict(Name, WeightStat, Type)

def LogTimeVaryingActivityForModule(self, data, Name, Type="TimeVaryingActivity", log="Data"):
    log = utils_torch.ParseLog(log)
    param = self.param
    data = utils_torch.ToNpArray(data)
    if hasattr(param, "FullName"):
        Name = param.FullName + "." + Name
    log.AddLogCache(Name, data, Type)

def LogActivityForModule(self, data, Name, Type="Activity", log="Data"):
    log = utils_torch.ParseLog(log)
    param = self.param
    data = utils_torch.ToNpArray(data)
    if hasattr(param, "FullName"):
        Name = param.FullName + "." + Name
    log.AddLogCache(Name, data, Type)

def LogForModule(self, data, Name, Type=None, log="Data"):
    log = utils_torch.ParseLog(log)
    param = self.param
    if hasattr(param, "FullName"):
        Name = param.FullName + "." + Name
    data = ProcessLogData(data)
    log.AddLog(Name, data, Type)

def LogWeightForModule(self, weights, Name="Weight", Type="Weight", log="Data"):
    log = utils_torch.ParseLog(log)
    param = self.param
    _weights = {}
    for name, weight in weights.items():
        _weights[name] = utils_torch.ToNpArray(weight)
    log.AddLogCache(Name, _weights, Type)

def LogFloatForModule(self, data, Name, Type="Float", log="Data"):
    log = utils_torch.ParseLog(log)
    param = self.param
    if isinstance(data, torch.Tensor):
        data = data.item()
    if hasattr(param, "FullName"):
        Name = param.FullName + "." + Name
    log.AddLog(Name, data, Type)

def LogLossForModule(self, loss, Name, Type="Loss", log="Data"):
    log = utils_torch.ParseLog(log)
    if isinstance(loss, torch.Tensor):
        data = loss.item()
    log.AddLog(Name, data, Type)

def LogAccuracyForSingleClassPrediction(ClassIndexPredicted, ClassIndexTruth, log):
    #log = utils_torch.ParseLog(log)
    ClassIndexPredicted = utils_torch.ToNpArray(ClassIndexPredicted)
    ClassIndexTruth = utils_torch.ToNpArray(ClassIndexTruth)
    NumCorrect, NumTotal = utils_torch.evaluate.CalculateAccuracyForSingelClassPrediction(ClassIndexPredicted, ClassIndexTruth)
    log.AddLogDict(
        "Accuracy",
        {
            "SampleNumTotal": NumTotal,
            "SampleNumCorrect": NumCorrect,
        }
    )
def LogCacheForModule(self, data, Name, Type=None, log="Data"):
    log = utils_torch.ParseLog(log)
    data = ProcessLogData(data)
    param = self.param
    if hasattr(param, "FullName"):
        Name = param.FullName + "." + Name
    log.AddLogCache(Name, data, Type)

def ProcessLogData(data):
    if isinstance(data, torch.Tensor):
        data = utils_torch.Tensor2NumpyOrFloat(data)
    return data


def GetTrainWeightForModule(self):
    return self.cache.TrainWeight

def PlotWeightForModule(self, SaveDir=None):
    if SaveDir is None:
        SaveDir = utils_torch.GetMainSaveDir() + "weights/"
    cache = self.cache
    if hasattr(self, "PlotSelfWeight"):
        self.PlotSelfWeight(SaveDir)
    if hasattr(cache, "Modules"):
        for ModuleName, Module in utils_torch.ListAttrsAndValues(cache.Modules):
            if hasattr(Module,"PlotWeight"):
                Module.PlotWeight(SaveDir)

def BuildModulesForModule(self):
    # initialize modules
    # for module in ListAttrs(param.modules):
    param = self.param
    cache = self.cache
    for Name, ModuleParam in ListAttrsAndValues(param.Modules, Exceptions=["__ResolveBase__"]):
        ModuleParam.Name = Name
        ModuleParam.FullName = param.FullName + "." + Name

        if not HasAttrs(ModuleParam, "Type"):
            if HasAttrs(ModuleParam, "Name"):
                SetAttrs(ModuleParam, "Type", GetAttrs(ModuleParam.Name))
            else:
                raise Exception()
        if ModuleParam.Type in ["Internal", "External"]:
            continue
        if cache.IsInit:
            Module = BuildModule(ModuleParam)
        else:
            Module = BuildModule(ModuleParam, LoadDir=cache.LoadDir)
        if isinstance(Module, nn.Module) and isinstance(self, nn.Module):
            self.add_module(Name, Module)
        setattr(cache.Modules, Name, Module)

def LoadFromFileForModule(self, LoadDir, Name, **kw):
    utils_torch.RemoveAttrIfExists(self, "cache")
    utils_torch.RemoveAttrIfExists(self, "param")
    utils_torch.RemoveAttrIfExists(self, "data")

    ParamPath = LoadDir + Name + ".param.jsonc"
    if utils_torch.FileExists(ParamPath):
        param = utils_torch.json.JsonFile2PyObj(ParamPath)
    else:
        param = utils_torch.EmptyPyObj()
    
    DataPath = LoadDir + Name + ".data"
    if utils_torch.FileExists(DataPath):
        data = utils_torch.json.DataFile2PyObj(DataPath)
    else:
        data = utils_torch.EmptyPyObj()

    cache = utils_torch.EmptyPyObj()
    cache.LoadDir = LoadDir


    param.cache.__object__ = self

    cache.Modules = utils_torch.EmptyPyObj()
    cache.Dynamics = utils_torch.EmptyPyObj()

    HasTensor = kw.setdefault("HasTensor", True)
    if HasTensor:
        cache.Tensors = []

    self.param = param
    self.data = data
    self.cache = cache
    self.Modules = cache.Modules
    self.Dynamics = cache.Dynamics

def InitModulesForModule(self):
    cache = self.cache
    for name, module in ListAttrsAndValues(cache.Modules):
        if hasattr(module, "InitFromParam"):
            module.InitFromParam(IsLoad=cache.IsLoad)
        else:
            if HasAttrs(module, "param.ClassPath"):
                Class = module.param.ClassPath
            else:
                Class = type(module)
            if not utils_torch.IsFunction(module):
                utils_torch.AddWarning(
                    "Module %s of class %s has not implemented InitFromParam method."%(name, Class)
                )
            if module is None:
                raise Exception(name)


def LoadFromParamForModule(self):
    self.InitFromParam(IsLoad=True)

def DoInitTasksForModule(self):
    param = self.param
    EnsureAttrs(param, "InitTasks", default=[])
    for Task in self.param.InitTasks:
        utils_torch.DoTask(Task, ObjCurrent=self.cache, ObjRoot=utils_torch.GetGlobalParam())

def Add2ObjRefListForParseRouters(ObjRef):
    GlobalParam = utils_torch.GetGlobalParam()
    if not hasattr(GlobalParam.cache, "AdditionalObjRefListForParseRouters"):
        GlobalParam.cache.AdditionalObjRefListForParseRouters = []
    GlobalParam.cache.AdditionalObjRefListForParseRouters.append(ObjRef)

def ParseRoutersForModule(self):
    GlobalParam = utils_torch.GetGlobalParam()
    param = self.param
    cache = self.cache
    for Name, RouterParam in ListAttrsAndValues(param.Dynamics, Exceptions=["__ResolveRef__", "__Entry__"]):
        if isinstance(RouterParam, str) and RouterParam in ["ClassMethod", "InternalMethod"]:
            setattr(cache.Dynamics, Name, getattr(self, Name))
            continue
        if cache.IsInit:
            utils_torch.router.ParseRouterStatic(RouterParam)
            setattr(RouterParam, "Name", param.FullName + "." + Name) # For Debug
        setattr(cache.Dynamics, Name, utils_torch.EmptyPyObj())
    
    ObjRefList = [
        cache.Modules, cache.Dynamics, cache,
        param, self, utils_torch.transform.Operators,
    ]
    if hasattr(GlobalParam.cache, "AdditionalObjRefListForParseRouters"):
        ObjRefList += GlobalParam.cache.AdditionalObjRefListForParseRouters
    for Name, RouterParam in ListAttrsAndValues(param.Dynamics, Exceptions=["__ResolveRef__", "__Entry__"]):
        if isinstance(RouterParam, str) and RouterParam in ["ClassMethod", "InternalMethod"]:
            continue
        getattr(cache.Dynamics, Name).FromPyObj(
            utils_torch.router.ParseRouterDynamic(
                RouterParam, 
                ObjRefList = ObjRefList, ObjRoot = utils_torch.GetGlobalParam(),
                InPlace=False
            )
        )
    return

def RegisterExternalMethodForModule(self, Name, Method):
    if not callable(Method):
        Method = utils_torch.parse.ResolveStr(Method)
    setattr(self, Name, Method)

def SaveForModule(self, SaveDir, Name=None, IsRoot=True):
    param = self.param
    data = self.data
    cache = self.cache

    if Name is None:
        SaveName = param.FullName
    else:
        SaveName = Name

    if IsRoot:
        SavePath = SaveDir + SaveName + ".param.jsonc"
        utils_torch.EnsureFileDir(SavePath)
        utils_torch.json.PyObj2JsonFile(param, SavePath)
    if not data.IsEmpty():
        data = utils_torch.parse.ApplyMethodOnPyObj(data, ToNpArrayIfIsTensor)
        PyObj2DataFile(data, SaveDir + SaveName + ".data")
    
    if hasattr(cache, "Modules"):
        for name, module in ListAttrsAndValues(cache.Modules):
            if HasAttrs(module, "Save"):
                module.Save(SaveDir, IsRoot=False)

def ToNpArrayIfIsTensor(data):
    if isinstance(data, torch.Tensor):
        return utils_torch.ToNpArray(data), False
    else:
        return data, True

def Interest():
    return

def SetMethodForModuleClass(Class, **kw):
    HasTensor = kw.setdefault("HasTensor", True)
    Class.Log = LogForModule
    Class.PlotWeight = PlotWeightForModule
    Class.SetFullName = SetFullNameForModule
    Class.GetFullName = GetFullNameForModule
    Class.RegisterExternalMethod = RegisterExternalMethodForModule
    Class.DoInitTasks = DoInitTasksForModule
    if HasTensor:
        Class.SetTensorLocation = AbstractModuleWithTensor.SetTensorLocation
        Class.GetTensorLocation = AbstractModuleWithTensor.GetTensorLocation
        if not hasattr(Class, "SetTrainWeight"):
            Class.SetTrainWeight = SetTrainWeightForModule
        if not hasattr(Class, "GetTrainWeight"):
            Class.GetTrainWeight = GetTrainWeightForModule
        if not hasattr(Class, "ClearTrainWeight"):
            Class.ClearTrainWeight = ClearTrainWeightForModule
        if not hasattr(Class, "SetPlotWeight"):
            Class.SetPlotWeight = SetPlotWeightForModule
        if not hasattr(Class, "GetPlotWeight"):
            Class.GetPlotWeight = GetPlotWeightForModule
        if not hasattr(Class, "ClearPlotWeight"):
            Class.ClearPlotWeight = ClearPlotWeightForModule
    if not hasattr(Class, "Save"):
        Class.Save = SaveForModule
    if not hasattr(Class, "LoadFromFile"):
        Class.LoadFromFile = LoadFromFileForModule
    Class.InitModules = InitModulesForModule
    Class.BuildModules = BuildModulesForModule
    if not hasattr(Class, "ParseRouters"):
        Class.ParseRouters = ParseRoutersForModule
    Class.LogTimeVaryingActivity = LogTimeVaryingActivityForModule
    Class.LogWeight = LogWeightForModule
    Class.LogWeightStat = LogWeightStatForModule
    Class.LogActivityStat = LogActivityStatForModule
    Class.LoadFromParam = LoadFromParamForModule
    Class.LogStat = LogStatForModule
    Class.LogCache = LogCacheForModule
    Class.LogDict = LogDictForModule
    Class.LogFloat = LogFloatForModule
    Class.LogLoss = LogLossForModule

def SetMethodForNonModelClass(Class, **kw):
    HasTensor = kw.setdefault("HasTensor", False)
    if not hasattr(Class, "SetFullName"):
        Class.SetFullName = SetFullNameForModule
    if not hasattr(Class, "Save"):
        Class.Save = SaveForModule
    Class.LoadFromParam = LoadFromParamForModule
    Class.RegisterExternalMethod = RegisterExternalMethodForModule
    if HasTensor:
        Class.SetTensorLocation = AbstractModuleWithTensor.SetTensorLocation
        Class.GetTensorLocation = AbstractModuleWithTensor.GetTensorLocation
    Class.InitModules = InitModulesForModule
    Class.BuildModules = BuildModulesForModule
    if not hasattr(Class, "ParseRouters"):
        Class.ParseRouters = ParseRoutersForModule

def ToFileForLogClassDataOnly(self, FilePath):
    if not FilePath.endswith(".data"):
        FilePath += ".data"
    utils_torch.files.PyObj2DataFile(self.data, FilePath)
    return self

def FromFileForLogClassDataOnly(self, FilePath):
    assert FilePath.endswith(".data")
    self.data = utils_torch.files.DataFile2PyObj(FilePath)
    return self

def ToFileForLogClass(self, SaveName=None, SaveDir=None):
    if SaveName is None:
        SaveName = self.param.FullName
    utils_torch.files.PyObj2DataFile(self.data,  SaveDir + SaveName + ".data")
    utils_torch.files.PyObj2JsonFile(self.param, SaveDir + SaveName + ".jsonc")
    return self

def FromFileForLogClass(self, SaveName, SaveDir):
    self.data  = utils_torch.files.DataFile2PyObj(SaveDir + SaveName + ".data")
    self.param = utils_torch.files.JsonFile2PyObj(SaveDir + SaveName + ".jsonc")
    self.cache = utils_torch.EmptyPyObj()
    return self

def SetMethodForLogClass(Class, **kw):
    SaveDataOnly = kw.setdefault("SaveDataOnly", False)
    MountLocation = kw.setdefault("MountLocation", "data")
    if SaveDataOnly:
        if not hasattr(Class, "ToFile"):
            Class.ToFile = ToFileForLogClassDataOnly
        if not hasattr(Class, "FromFile"):
            Class.FromFile = FromFileForLogClassDataOnly
    else:
        if not hasattr(Class, "ToFile"):
            Class.ToFile = ToFileForLogClass
        if not hasattr(Class, "FromFile"):
            Class.FromFile = FromFileForLogClass
    # if MountLocation in ["Data", "data"]:
    #     Class.SetEpochBatchIndex = SetEpochBatchIndexForModuleData
    # elif MountLocation in ["Cache", "cache"]:
    #     Class.SetEpochBatchIndex = SetEpochBatchIndexForModuleCache
    SetEpochBatchMethodForModule(Class, **kw)


# def SetEpochIndexForModule(self, EpochIndex):
#     self.cache.EpochIndex = EpochIndex

# def SetBatchIndexForModule(self, BatchIndex):
#     self.cache.BatchIndex = BatchIndex

# def SetEpochNumForModule(self, EpochNum):
#     self.cache.EpochNum = EpochNum

# def SetBatchNumForModule(self, BatchNum):
#     self.cache.BatchNum = BatchNum

def ParseParamForModule(self, **kw):
    GlobalParam = utils_torch.GetGlobalParam()
    utils_torch.parse.ParsePyObjStatic(self.param, ObjCurrent=self.param, ObjRoot=GlobalParam)

from utils_torch.transform.MLP import MLP
from utils_torch.transform.SignalTrafficNodes import SerialReceiver
from utils_torch.transform.SignalTrafficNodes import SerialSender
from utils_torch.transform.SignalTrafficNodes import SignalHolder
from utils_torch.transform.LambdaLayer import LambdaLayer
from utils_torch.transform.RecurrentLIFLayer import RecurrentLIFLayer
from utils_torch.transform.NoiseGenerator import NoiseGenerator
from utils_torch.transform.Bias import Bias
import utils_torch.transform.Operators as Operators
from utils_torch.transform.SingleLayer import SingleLayer
from utils_torch.transform.NonLinearLayer import NonLinearLayer
from utils_torch.transform.LinearLayer import LinearLayer
from utils_torch.module.AbstractModules import SetEpochBatchMethodForModule

ModuleDict = {
    "MLP": MLP, "mlp": MLP,
    "SerialReceiver": SerialReceiver, "SerialSend": SerialSender, "SignalHolder": SignalHolder,
    "SingalLayer": SingleLayer, "LinearLayer": LinearLayer, "NonLinearLayer": NonLinearLayer,
    "LambbdaLayer": LambdaLayer,
}