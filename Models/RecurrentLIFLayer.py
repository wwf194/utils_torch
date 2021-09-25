import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

DefaultRoutings = [
    "&GetBias |--> bias",
    "&GetNoise |--> noise",
    "hiddenState, input, noise, bias |--> &Add |--> inputTotal",
    "inputTotal, cellState |--> &ProcessInputTotalAndcellState |--> cellState",
    "cellState |--> &NonLinear |--> hiddenState",
    "hiddenState |--> &HiddenStateTransform |--> hiddenState",
    "cellState |--> &CellStateDecay |--> cellState"
]

class RecurrentLIFLayer(nn.Module):
    def __init__(self, param=None):
        super(RecurrentLIFLayer, self).__init__()
        if param is not None:
            self.param = param
            self.data = utils_torch.EmptyPyObj()
            self.cache = utils_torch.EmptyPyObj()
    def InitFromParam(self, param=None):
        if param is None:
            param = self.param
            data = self.data
            cache = self.cache
        else:
            self.param = param
            self.data = utils_torch.EmptyPyObj()
            self.cache = utils_torch.EmptyPyObj()
        
        self.cache.Modules = utils_torch.EmptyPyObj()
        
        EnsureAttrs(param, "IsExciInhi", default=False)
        cache.ParamIndices = []
        self.BuildModules()
        self.InitModules()
        self.SetInternalMethods()
        self.ParseRouters()

    def BuildModules(self):
        param = self.param
        cache = self.cache
        for Name, ModuleParam in ListAttrsAndValues(param.Modules, Exceptions=["__ResolveRef__"]):
            # if Name in ["GetBias"]:
            #     print("aaa")
            if hasattr(ModuleParam, "Type") and ModuleParam.Type in ["Internal"]:
                continue
            setattr(ModuleParam, "Name", Name)
            Module = utils_torch.model.BuildModule(ModuleParam)
            setattr(cache.Modules, Name, Module)
            if isinstance(Module, nn.Module):
                self.add_module(Name, Module)
    def InitModules(self):
        param = self.param
        cache = self.cache
        for Module in ListValues(cache.Modules):
            if hasattr(Module, "InitFromParam"):
                Module.InitFromParam()
            else:
                utils_torch.AddWarning("Module %s has not implemented InitFromParam method."%Module)
    def SetInternalMethods(self):
        param = self.param
        cache = self.cache
        Modules = cache.Modules
        if param.IsExciInhi:
            if not (HasAttrs(param, "TimeConst.Excitatory") and HasAttrs(param, "TimeConst.Inhibitory")):
                EnsureAttrs(param, "TimeConst", default=0.1)
                SetAttrs(param, "TimeConst.Excitatory", param.TimeConst)
                SetAttrs(param, "TimeConst.Inhibitory", param.TimeConst)
            ExciNeuronsNum = param.Neurons.Excitatory.Num
            InhiNeuronsNum = param.Neuorns.Inhibitory.Num
            if param.TimeConst.Excitatory==param.TimeConst.Inhibitory:
                TimeConst = param.TimeConst.Excitatory
                Modules.CellStateDecay = lambda CellState: TimeConst * CellState
                Modules.ProcessTotalInput = lambda TotalInput: (1.0 - TimeConst) * TotalInput
                Modules.ProcessCellStateAndTotalInput = lambda CellState, TotalInput: \
                    CellState + Modules.ProcessTotalInput(TotalInput)
            else:
                TimeConstExci = param.TimeConst.Excitatory
                TimeConstInhi = param.TimeConst.Inhibitory
                if not (0.0 <= TimeConstExci <= 1.0 and 0.0 <= TimeConstExci <= 1.0):
                    raise Exception()
                
                Modules.CellStateDecay = lambda CellState: \
                    torch.concatenate([
                            CellState[:, :ExciNeuronsNum] * TimeConstExci, 
                            CellState[:, ExciNeuronsNum:] * TimeConstInhi
                        ], 
                        axis=1
                    )
                Modules.ProcessTotalInput = lambda TotalInput: \
                    torch.concatenate([
                            TotalInput[:, :ExciNeuronsNum] * (1.0 - TimeConstExci),
                            TotalInput[:, ExciNeuronsNum:] * (1.0 - TimeConstInhi),
                        ], 
                        axis=1
                    )
                Modules.ProcessCellStateAndTotalInput = lambda CellState, TotalInput: \
                    CellState + Modules.ProcessTotalInput(TotalInput)
        else:
            EnsureAttrs(param, "TimeConst", default=0.1)
            TimeConst = GetAttrs(param.TimeConst)
            if not 0.0 <= TimeConst <= 1.0:
                raise Exception()
            Modules.CellStateDecay = lambda CellState: TimeConst * CellState
            Modules.ProcessTotalInput = lambda TotalInput: (1.0 - TimeConst) * TotalInput
            Modules.ProcessCellStateAndTotalInput = lambda CellState, TotalInput: \
                CellState + Modules.ProcessTotalInput(TotalInput)
    def ParseRouters(self):
        param = self.param
        cache = self.cache
        cache.Dynamics = utils_torch.EmptyPyObj()
        for Name, RouterParam in ListAttrsAndValues(param.Dynamics, Exceptions=["__ResolveRef__", "__Entry__"]):
            utils_torch.router.ParseRouterStatic(RouterParam)
        for Name, RouterParam in ListAttrsAndValues(param.Dynamics, Exceptions=["__ResolveRef__", "__Entry__"]):
            Router = utils_torch.router.ParseRouterDynamic(RouterParam, 
                ObjRefList=[cache.Modules, cache.Dynamics, cache, param, self, utils_torch.Models.Operators])
            setattr(cache.Dynamics, Name, Router)
        if not HasAttrs(param.Dynamics, "__Entry__"):
            SetAttrs(param, "Dynamics.__Entry__", "&Dynamics.%s"%ListAttrs(param.Dynamics)[0])
        cache.Dynamics.__Entry__ = utils_torch.parse.Resolve(param.Dynamics.__Entry__, ObjRefList=[cache, self])
        return
    def forward(self, CellState, RecurrentInput, Input):
        cache = self.cache
        return utils_torch.CallGraph(cache.Dynamics.__Entry__, [CellState, RecurrentInput, Input])
    def SetTensorLocation(self, Location):
        cache = self.cache
        utils_torch.model.SetTensorLocationForLeafModel(self, Location)
        for Name, Module in ListAttrsAndValues(cache.Modules):
            if hasattr(Module, "SetTensorLocation"):
                Module.SetTensorLocation(Location)    
    def GetTensorLocation(self):
        return self.cache.TensorLocation
    def GetTrainWeight(self):
        return self.cache.TrainWeight
    def SetTrainWeight(self):
        return utils_torch.model.SetTrainWeightForModel(self)
    def ClearTrainWeight(self):
        cache = self.cache
        if hasattr(cache, "TrainWeight"):
            delattr(cache, "TrainWeight")
__MainClass__ = RecurrentLIFLayer