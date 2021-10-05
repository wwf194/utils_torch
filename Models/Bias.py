import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

class Bias(nn.Module):
    def __init__(self, param=None):
        super(Bias, self).__init__()
        if param is not None:
            self.param = param
            self.cache = utils_torch.EmptyPyObj()
            self.data = utils_torch.EmptyPyObj()
    def InitFromParam(self):
        param = self.param
        data = self.data
        cache = self.cache
        cache.Tensors =[]
        data.Bias = torch.nn.Parameter(torch.zeros(param.Size))
        cache.Tensors.append([data, "Bias", data.Bias])
    def forward(self):
        return self.data.Bias
    def SetTensorLocation(self, Location):
        utils_torch.model.SetTensorLocationForLeafModel(self, Location)
        return
    def GetTensorLocation(self):
        return self.cache.TensorLocation
__MainClass__ = Bias
utils_torch.model.SetMethodForModelClass(__MainClass__)