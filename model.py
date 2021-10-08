
from typing import Set
from cv2 import Tonemap
import torch
from torch._C import ThroughputBenchmark
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import utils_torch
#from utils_torch.utils import EnsurePath, Getargs, Getname, Getname_args, GetFromDict, search_dict, contain, contain_all, Getrow_col, prep_title
from utils_torch.json import *
from utils_torch.attrs import *
from utils_torch.LRSchedulers import LinearLR

def BuildModule(param, **kw):
    if not HasAttrs(param, "Type"):
        if HasAttrs(param, "Name"):
            SetAttrs(param, "Type", GetAttrs(param.Name))
        else:
            raise Exception()
    if param.Type in ["LinearLayer"]:
        return utils_torch.Models.LinearLayer(param, **kw)
    elif param.Type in ["NonLinearLayer"]:
        return utils_torch.Models.NonLinearLayer(param, **kw)
    elif param.Type in ["MLP", "MultiLayerPerceptron", "mlp"]:
        return utils_torch.Models.MLP(param, **kw)
    elif param.Type in ["SerialReceiver"]:
        return utils_torch.Models.SerialReceiver(param, **kw)
    elif param.Type in ["SerialSender"]:
        return utils_torch.Models.SerialSender(param, **kw)
    elif param.Type in ["Lambda", "LambdaLayer"]:
        return utils_torch.Models.LambdaLayer(param, **kw)
    elif param.Type in ["RecurrentLIFLayer"]:
        return utils_torch.Models.RecurrentLIFLayer(param, **kw)
    elif param.Type in ["NoiseGenerator"]:
        return utils_torch.Models.NoiseGenerator(param, **kw)
    elif param.Type in ["Bias"]:
        return utils_torch.Models.Bias(param, **kw)
    elif param.Type in ["NonLinear"]:
        return GetNonLinearMethod(param, **kw)
    elif param.Type in ["L2Loss"]:
        return utils_torch.Models.L2Loss(param, **kw)
    elif param.Type in ["MSE", "MeanSquareError"]:
        return utils_torch.Models.Loss.GetLossMethod(param, **kw)
    elif param.Type in ["GradientDescend"]:
        return utils_torch.Models.Operators.GradientDescend(param, **kw)
    elif param.Type in ["Internal"]:
        utils_torch.AddWarning("utils_torch.model.BuildModule does not build Module of type Internal.")
        return None
    elif utils_torch.Models.Operators.IsLegalType(param.Type):
        return utils_torch.Models.Operators.BuildModule(param, **kw)
    elif hasattr(param, "ModulePath"):
        Module = utils_torch.ImportModule(param.ModulePath)
        Obj = Module.__MainClass__(param, **kw)
        return Obj
    else:
        raise Exception("BuildModule: No such module: %s"%param.Type)

def CalculateWeightChangeRatio(Weight, WeightChange):
    Weight = utils_torch.ToNpArray(Weight)
    WeightChange = utils_torch.ToNpArray(WeightChange)
    WeightChangeRatio = np.sum(np.abs(WeightChange)) / np.sum(np.abs(Weight))
    return WeightChangeRatio

def ListParameter(model):
    for name, param in model.named_parameters():
        utils_torch.AddLog("%s: Shape: %s"%(name, param.size()))

def CreateSelfConnectionMask(Size):
    return torch.from_numpy(np.ones(Size, Size) - np.eye(Size))

def CreateExcitatoryInhibitoryMask(InputNum, OutputNum, ExcitatoryNum, InhibitoryNum=None):
    # Assumed weight matrix shape: [InputNum, OutputNum]
    if InhibitoryNum is None:
        InhibitoryNum = InputNum - ExcitatoryNum
    else:
        if InputNum != ExcitatoryNum + InhibitoryNum:
            raise Exception("GetExcitatoryInhibitoryMask: InputNum==ExcitatoryNum + InhibitoryNum must be satisfied.")

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

def GetNonLinearMethod(param):
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
    elif Type in ["tanh"]:
        if param.Coefficient==1.0:
            return F.tanh
        else:
            return lambda x:param.Coefficient * F.tanh(x)       
    elif Type in ["sigmoid"]:
        if param.Coefficient==1.0:
            return F.tanh
        else:
            return lambda x:param.Coefficient * F.tanh(x)         
    else:
        raise Exception("GetNonLinearMethod: Invalid nonlinear function Type: %s"%param.Type)

Getactivation_function = GetNonLinearMethod

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

def CreateWeight2D(param, DataType=torch.float32):
    Init = param.Init
    if Init.Method in ["kaiming", "he"]:
        Init.Method = "kaiming"
        EnsureAttrs(Init, "Mode", default="In")
        EnsureAttrs(Init, "Distribution", default="uniform")
        EnsureAttrs(Init, "Coefficient", default=1.0)
        if Init.Mode in ["In"]:
            if Init.Distribution in ["uniform"]:
                Init.Range = [ - Init.Coefficient * 6 ** 0.5 / param.Size[0] ** 0.5,
                    Init.Coefficient * 6 ** 0.5 / param.Size[0] ** 0.5
                ]
                weight = np.random.uniform(*Init.Range, tuple(param.Size))
            elif Init.Distribution in ["uniform+"]:
                Init.Range = [
                    0.0,
                    2.0 * Init.Coefficient * 6 ** 0.5 / param.Size[0] ** 0.5
                ]
                weight = np.random.uniform(*Init.Range, tuple(param.Size))
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

def build_scheduler(dict_, optimizer, load=False, verbose=True):
    #lr_decay = dict['lr_decay']
    scheduler_dict = dict_['scheduler']
    scheduler_Type = search_dict(scheduler_dict, ['Type', 'Method'], default='None', write_default=True)
    if verbose:
        print('build_scheduler: scheduler_Type: %s'%scheduler_Type)
    if scheduler_Type is None or scheduler_Type in ['None', 'none']:
        scheduler = None
        #update_lr = update_lr_none
    elif scheduler_Type in ['exp']:
        decay = search_dict(scheduler_dict, ['decay', 'coeff'], default=0.98, write_default=True, write_default_dict='decay')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
        #update_lr = update_lr_
    elif scheduler_Type in ['stepLR', 'exp_interval']:
        decay = search_dict(scheduler_dict, ['decay', 'coeff'], default=0.98, write_default=True, write_default_key='decay')
        step_Size = search_dict(scheduler_dict, ['interval', 'step_Size'], default=0.98, write_default=True, write_default_key='decay')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, step_Size=step_Size, gamma=decay)
        #update_lr = update_lr_
    elif scheduler_Type in ['Linear', 'linear']:
        milestones = search_dict(scheduler_dict, ['milestones'], throw_none_error=True)
        scheduler = LinearLR(optimizer, milestones=milestones, epoch_num=dict_['epoch_num'])
        #update_lr = update_lr_
    else:
        raise Exception('build_scheduler: Invalid lr decay Method: '+str(scheduler_Type))
    return scheduler

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

def SetTensorLocationForModel(self, Location):
    cache = self.cache
    cache.TensorLocation = Location
    if hasattr(cache, "Tensors"):
        for ParamIndex in cache.Tensors:
            setattr(ParamIndex[0], ParamIndex[1], ParamIndex[2].to(Location).detach().requires_grad_(True))

    if hasattr(cache, "Modules"):
        for name, module in ListAttrsAndValues(cache.Modules):
            if hasattr(module, "SetTensorLocation"):
                module.SetTensorLocation(Location)
            else:
                if isinstance(module, nn.Module):
                    utils_torch.AddWarning("%s is an instance of nn.Module, but has not implemented SetTensorLocation method."%name)

def SetTrainWeightForModel(self):
    ClearTrainWeightForModel(self)
    cache = self.cache
    cache.TrainWeight = {}
    if hasattr(cache, "Modules"):
        for ModuleName, Module in utils_torch.ListAttrsAndValues(cache.Modules):
            if hasattr(Module,"SetTrainWeight"):
                TrainWeight = Module.SetTrainWeight()
                for name, weight in TrainWeight.items():
                    cache.TrainWeight[ModuleName + "." + name] = weight
            else:
                if isinstance(Module, nn.Module):
                    utils_torch.AddWarning("Module %s is instance of nn.Module, but has not implemented GetTrainWeight method."%Module)
        return cache.TrainWeight
    else:
        return {}

def GetPlotWeightForModel(self):
    cache = self.cache
    if not hasattr(cache, "PlotWeight"):
        self.SetPlotWeight()
    weights = {}
    for name, method in cache.PlotWeight.items():
        weights[name] = method()
    return weights

def SetPlotWeightForModel(self):
    ClearPlotWeightForModel(self)
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

def ClearPlotWeightForModel(self):
    cache = self.cache
    if hasattr(cache, "PlotWeight"):
        delattr(cache, "PlotWeight")

def ClearTrainWeightForModel(self):
    cache = self.cache
    if hasattr(cache, "TrainWeight"):
        delattr(cache, "TrainWeight")

def SetLoggerForModel(self, logger):
    cache = self.cache
    cache.Logger = logger
    if hasattr(cache, "Modules"):   
        for Name, Module in ListAttrsAndValues(self.cache.Modules):
            if hasattr(Module, "SetLogger"):
                Module.SetLogger(utils_torch.log.DataLogger().SetParent(logger, prefix=Name + "."))

def SetFullNameForModel(self, FullName):
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
def GetLoggerForModel(self):
    cache = self.cache
    if hasattr(cache, "Logger"):
        return cache.Logger
    else:
        return None

def PlotWeight(weight, Name, Save=True, SavePath="./weight.png"):
    weight = utils_torch.ToNpArray(weight)
    DimensionNum = len(weight.shape)

def PlotActivity(activity, Name, Save=True, SavePath="./weight.png"):
    # @param activity: [BatchSize, StepNum, NeuronNum]
    activity = utils_torch.ToNpArray(activity)
    return

def InitForModel(self, param=None, data=None, DefaultFullName="Unnamed", ClassPath=None, **kw):
    LoadDir = kw.get("LoadDir")
    if param is None:
        param = utils_torch.EmptyPyObj()
    
    if not hasattr(param, "FullName"):
        param.FullName = DefaultFullName
    param.cache.__object__ = self

    if data is None:
        data = utils_torch.EmptyPyObj()
        if LoadDir is not None:
            DataPath = LoadDir + param.FullName + ".data"
            if utils_torch.FileExists(DataPath):
                data = utils_torch.json.DataFile2PyObj(DataPath)

    cache = utils_torch.EmptyPyObj()
    if LoadDir is not None:
        cache.LoadDir = LoadDir
    else:
        cache.LoadDir = None
    if ClassPath is not None:
        param.ClassPath = ClassPath
    
    cache.Modules = utils_torch.EmptyPyObj()
    cache.Dynamics = utils_torch.EmptyPyObj()
    
    self.param = param
    self.data = data
    self.cache = cache
    self.Modules = cache.Modules
    self.Dynamics = cache.Dynamics

def LogStatisticsForModel(self, data, Name, Type="Statistics"):
    param = self.param
    if hasattr(param, "FullName"):
        Name = param.FullName + "." + Name
    utils_torch.GetDataLogger().AddLogStatistics(Name, data, Type)

def LogTimeVaryingActivityForModel(self, data, Name, Type="TimeVaryingActivity"):
    param = self.param
    data = utils_torch.ToNpArray(data)
    if hasattr(param, "FullName"):
        Name = param.FullName + "." + Name
    utils_torch.GetDataLogger().AddLogCache(Name, data, Type)

def LogForModel(self, data, Name, Type=None):
    param = self.param
    if hasattr(param, "FullName"):
        Name = param.FullName + "." + Name
    data = ProcessLogData(data)
    utils_torch.GetDataLogger().AddLog(Name, data, Type)

def LogWeightForModel(self, weights, Name="Weight", Type="Weight"):
    param = self.param
    _weights = {}
    for name, weight in weights.items():
        _weights[name] = utils_torch.ToNpArray(weight)
    utils_torch.GetDataLogger().AddLogCache(Name, _weights, Type)

def LogFloatForModel(self, data, Name, Type="Float"):
    param = self.param
    if isinstance(data, torch.Tensor):
        data = data.item()
    if hasattr(param, "FullName"):
        Name = param.FullName + "." + Name
    utils_torch.GetDataLogger().AddLog(Name, data, Type)

def LogCacheForModel(self, data, Name, Type=None):
    data = ProcessLogData(data)
    param = self.param
    if hasattr(param, "FullName"):
        Name = param.FullName + "." + Name
    utils_torch.GetDataLogger().AddLogCache(Name, data, Type)

def ProcessLogData(data):
    if isinstance(data, torch.Tensor):
        data = utils_torch.Tensor2NumpyOrFloat(data)
    return data

def GetTensorLocationForModel(self):
    return self.cache.TensorLocation

def GetTrainWeightForModel(self):
    return self.cache.TrainWeight


def PlotWeightForModel(self, SaveDir=None):
    if SaveDir is None:
        SaveDir = utils_torch.GetSaveDir() + "weights/"
    cache = self.cache
    if hasattr(self, "PlotSelfWeight"):
        self.PlotSelfWeight(SaveDir)
    if hasattr(cache, "Modules"):
        for ModuleName, Module in utils_torch.ListAttrsAndValues(cache.Modules):
            if hasattr(Module,"PlotWeight"):
                Module.PlotWeight(SaveDir)

def BuildModulesForModel(self):
    # initialize modules
    # for module in ListAttrs(param.modules):
    param = self.param
    cache = self.cache
    for Name, ModuleParam in ListAttrsAndValues(param.Modules, Exceptions=["__ResolveBase__"]):
        ModuleParam.Name = Name
        ModuleParam.FullName = param.FullName + "." + Name
        if cache.IsInit:
            Module = BuildModule(ModuleParam)
        else:
            Module = BuildModule(ModuleParam, LoadDir=cache.LoadDir)
        if isinstance(Module, nn.Module) and isinstance(self, nn.Module):
            self.add_module(Name, Module)
        setattr(cache.Modules, Name, Module)

def InitModulesForModel(self):
    cache = self.cache
    for name, module in ListAttrsAndValues(cache.Modules):
        if hasattr(module, "InitFromParam"):
            module.InitFromParam()
        else:
            utils_torch.AddWarning("Module %s has not implemented InitFromParam method."%name)

def LoadFromParamForModel(self):
    self.InitFromParam(IsLoad=True)

def ParseRoutersForModel(self):
    param = self.param
    cache = self.cache
    cache.Dynamics = utils_torch.EmptyPyObj()
    for Name, RouterParam in ListAttrsAndValues(param.Dynamics, Exceptions=["__ResolveRef__", "__Entry__"]):
        utils_torch.router.ParseRouterStatic(RouterParam)
        setattr(RouterParam, "Name", param.FullName + "." + Name) # For Debug
        setattr(cache.Dynamics, Name, utils_torch.EmptyPyObj())
    for Name, RouterParam in ListAttrsAndValues(param.Dynamics, Exceptions=["__ResolveRef__", "__Entry__"]):
        getattr(cache.Dynamics, Name).FromPyObj(
            utils_torch.router.ParseRouterDynamic(
                RouterParam, 
                ObjRefList=[cache.Modules, cache.Dynamics, cache,
                    param, self, utils_torch.Models.Operators
                ],
                InPlace=False
            )
        )
    return

def SaveForModel(self, SaveDir, Name=None, IsRoot=True):
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

def SetMethodForModelClass(Class):
    Class.LogStatistics = LogStatisticsForModel
    Class.LogCache = LogCacheForModel
    Class.LogFloat = LogFloatForModel
    Class.Log = LogForModel
    Class.PlotWeight = PlotWeightForModel
    Class.SetFullName = SetFullNameForModel
    Class.BuildModules = BuildModulesForModel
    Class.InitModules = InitModulesForModel
    Class.SetTensorLocation = SetTensorLocationForModel
    Class.GetTensorLocation = GetTensorLocationForModel
    if not hasattr(Class, "SetTrainWeight"):
        Class.SetTrainWeight = SetTrainWeightForModel
    if not hasattr(Class, "GetTrainWeight"):
        Class.GetTrainWeight = GetTrainWeightForModel
    if not hasattr(Class, "ClearTrainWeight"):
        Class.ClearTrainWeight = ClearTrainWeightForModel
    if not hasattr(Class, "SetPlotWeight"):
        Class.SetPlotWeight = SetPlotWeightForModel
    if not hasattr(Class, "GetPlotWeight"):
        Class.GetPlotWeight = GetPlotWeightForModel
    if not hasattr(Class, "ClearPlotWeight"):
        Class.ClearPlotWeight = ClearPlotWeightForModel
    if not hasattr(Class, "Save"):
        Class.Save = SaveForModel
    if not hasattr(Class, "ParseRouters"):
        Class.ParseRouters = ParseRoutersForModel
    Class.LogTimeVaryingActivity = LogTimeVaryingActivityForModel
    Class.LogWeight = LogWeightForModel
    Class.LoadFromParam = LoadFromParamForModel

def SetMethodForWorldClass(Class):
    if not hasattr(Class, "SetFullName"):
        Class.SetFullName = SetFullNameForModel
    if not hasattr(Class, "Save"):
        Class.Save = SaveForModel
    Class.LoadFromParam = LoadFromParamForModel