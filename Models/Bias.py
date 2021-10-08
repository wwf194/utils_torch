import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

class Bias(nn.Module):
    def __init__(self, param=None, data=None):
        super(Bias, self).__init__()
        utils_torch.model.InitForModel(self, param, data, ClassPath="utils_torch.Models.Bias")
    def InitFromParam(self):
        param = self.param
        data = self.data
        cache = self.cache
        cache.Tensors =[]
        data.Bias = torch.nn.Parameter(torch.zeros(param.Size))
        cache.Tensors.append([data, "Bias", data.Bias])
    def forward(self):
        return self.data.Bias

__MainClass__ = Bias
utils_torch.model.SetMethodForModelClass(__MainClass__)