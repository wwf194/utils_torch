import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.model import *
from utils_torch.utils import *
from utils_torch.utils import HasAttrs, EnsureAttrs, MatchAttrs, StackFunction, SetAttrs
from utils_torch.model import GetNonLinearMethod, GetConstraintFunction, CreateSelfConnectionMask, CreateExcitatoryInhibitoryMask, CreateWeight2D

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
            self.data = utils_torch.json.EmptyPyObj()
            self.cache = utils_torch.json.EmptyPyObj()
    def InitFromParam(self, param=None):
        if param is None:
            param = self.param
            data = self.data
            cache = self.cache
        else:
            self.param = param
            self.data = utils_torch.json.EmptyPyObj()
            self.cache = utils_torch.json.EmptyPyObj()
        
        self.cache.Modules = utils_torch.json.EmptyPyObj()
        
        EnsureAttrs(param, "IsExciInhi", default=False)
        cache.ParamIndices = []
        self.BuildModules()
        self.InitModules()
        self.SetInternalMethods()
        self.ParseRouter()

    def BuildModules(self):
        param = self.param
        cache = self.cache
        for Name, ModuleParam in ListAttrsAndValues(param.Modules, Exceptions=["__ResolveRef__"]):
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
    def ParseRouter(self):
        param = self.param
        cache = self.cache
        utils_torch.router.ParseRouterStatic(param.Router)
        cache.Router = utils_torch.router.ParseRouterDynamic(param.Router, 
            ObjRefList=[self.cache.Modules, self.cache, self.param, self, utils_torch.Models.Operators])
        return
    def forward(self, CellState, RecurrentInput, Input):
        cache = self.cache
        return utils_torch.CallGraph(cache.Router, [CellState, RecurrentInput, Input])
    def SetTensorLocation(self, Location):
        cache = self.cache
        cache.TensorLocation = Location
        for ParamIndex in cache.ParamIndices:
            setattr(ParamIndex[0], ParamIndex[1], ParamIndex[2].to(Location))
        for Name, Module in ListAttrsAndValues(cache.Modules):
            if hasattr(Module, "SetTensorLocation"):
                Module.SetTensorLocation(Location)    
    def GetTensorLocation(self):
        return self.cache.TensorLocation

__MainClass__ = RecurrentLIFLayer