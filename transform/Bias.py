import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

from utils_torch.module.AbstractModules import AbstractModuleWithTensor
class Bias(AbstractModuleWithTensor):
    def __init__(self, param=None, data=None, **kw):
        super(Bias, self).__init__()
        utils_torch.transform.InitForModule(self, param, data, ClassPath="utils_torch.transform.Bias", **kw)
    def InitFromParam(self, IsLoad=False):
        utils_torch.transform.InitFromParamForModule(self, IsLoad)
        param = self.param
        data = self.data
        cache = self.cache
        if cache.IsInit:
            data.Bias = torch.nn.Parameter(torch.zeros(param.Size))
        else:
            data.Bias = utils_torch.ToTorchTensor(data.Bias)
        cache.Tensors.append([data, "Bias", data.Bias])
    def forward(self):
        return self.data.Bias

__MainClass__ = Bias
# utils_torch.transform.SetMethodForModuleClass(__MainClass__)