import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

from utils_torch.transform import AbstractTransformWithTensor
class Bias(AbstractTransformWithTensor):
    # def __init__(self, param=None, data=None, **kw):
    #     super(Bias, self).__init__()
    #     self.InitModule(self, param, data, ClassPath="utils_torch.transform.Bias", **kw)
    def __init__(self, **kw):
        super().__init__(**kw)
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
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
# utils_torch.transform.SetMethodForTransformModule(__MainClass__)