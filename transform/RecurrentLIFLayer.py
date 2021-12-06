import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

DefaultRoutings = [
    "&GetBias |--> bias",
    "&GetNoise |--> noise",
    "hiddenState, input, noise, bias |--> &Add |--> inputTotal",
    "inputTotal, membranePotential |--> &ProcessInputTotalAndmembranePotential |--> membranePotential",
    "membranePotential |--> &NonLinear |--> hiddenState",
    "hiddenState |--> &HiddenStateTransform |--> hiddenState",
    "membranePotential |--> &MembranePotentialDecay |--> membranePotential"
]

from utils_torch.transform import AbstractTransformWithTensor
class RecurrentLIFLayer(AbstractTransformWithTensor):
    # def __init__(self, param=None, data=None, **kw):
    #     super(RecurrentLIFLayer, self).__init__()
    #     self.InitModule(self, param, data, ClassPath="utils_torch.transform.RecurrentLIFLayer", **kw)
    def __init__(self, **kw):
        super().__init__(**kw)
        return
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        param = self.param
        data = self.data
        cache = self.cache
        EnsureAttrs(param, "IsExciInhi", default=False)
        
        self.BuildModules()
        self.InitModules()
        self.SetInternalMethods()
        self.ParseRouters()

    def SetInternalMethods(self):
        param = self.param
        cache = self.cache
        Modules = cache.Modules
        if param.IsExciInhi:
            if cache.IsInit:
                if not (HasAttrs(param, "TimeConst.Excitatory") and HasAttrs(param, "TimeConst.Inhibitory")):
                    EnsureAttrs(param, "TimeConst", default=0.1)
                    SetAttrs(param, "TimeConst.Excitatory", GetAttrs(param.TimeConst))
                    SetAttrs(param, "TimeConst.Inhibitory", GetAttrs(param.TimeConst))
                    utils_torch.transform.ParseExciInhiNum(param.Neurons)
            ExciNeuronsNum = param.Neurons.Excitatory.Num
            InhiNeuronsNum = param.Neurons.Inhibitory.Num
            #ExciNeuronsNum = 80

            if param.TimeConst.Excitatory==param.TimeConst.Inhibitory:
                TimeConst = param.TimeConst.Excitatory
                Modules.MembranePotentialDecay = lambda MembranePotential: (1.0 - TimeConst) * MembranePotential
                Modules.ProcessTotalInput = \
                    lambda TotalInput: TimeConst * TotalInput
                Modules.ProcessMembranePotentialAndTotalInput = \
                    lambda MembranePotential, TotalInput: \
                    MembranePotential + Modules.ProcessTotalInput(TotalInput)
            else:
                TimeConstExci = param.TimeConst.Excitatory
                TimeConstInhi = param.TimeConst.Inhibitory
                if not (0.0 <= TimeConstExci <= 1.0 and 0.0 <= TimeConstExci <= 1.0):
                    raise Exception()
                Modules.MembranePotentialDecay = lambda MembranePotential: \
                    torch.concatenate([
                            MembranePotential[:, :ExciNeuronsNum] * (1.0 - TimeConstExci), 
                            MembranePotential[:, ExciNeuronsNum:] * (1.0 - TimeConstInhi),
                        ], 
                        axis=1
                    )
                Modules.ProcessTotalInput = lambda TotalInput: \
                    torch.concatenate([
                            TotalInput[:, :ExciNeuronsNum] * TimeConstExci,
                            TotalInput[:, ExciNeuronsNum:] * TimeConstInhi,
                        ], 
                        axis=1
                    )
                Modules.ProcessMembranePotentialAndTotalInput = lambda MembranePotential, TotalInput: \
                    MembranePotential + Modules.ProcessTotalInput(TotalInput)
        else:
            if cache.IsInit:
                EnsureAttrs(param, "TimeConst", default=0.1)
            TimeConst = GetAttrs(param.TimeConst)
            if not 0.0 <= TimeConst <= 1.0:
                raise Exception()
            Modules.MembranePotentialDecay = lambda MembranePotential: (1.0 - TimeConst) * MembranePotential
            Modules.ProcessTotalInput = lambda TotalInput: TimeConst * TotalInput
            Modules.ProcessMembranePotentialAndTotalInput = lambda MembranePotential, TotalInput: \
                MembranePotential + Modules.ProcessTotalInput(TotalInput)
    def forward(self, MembranePotential, RecurrentInput, Input):
        cache = self.cache
        return utils_torch.CallGraph(cache.Dynamics.Main, [MembranePotential, RecurrentInput, Input])  
__MainClass__ = RecurrentLIFLayer
# utils_torch.transform.SetMethodForTransformModule(__MainClass__)