import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.attrs import *

class LambdaLayer(nn.Module):
    def __init__(self, param=None, data=None):
        super(LambdaLayer, self).__init__()
        utils_torch.model.InitForModel(self, param, data, ClassPath="utils_torch.Models.LambdaLayer")
    def InitFromParam(self):
        param = self.param
        self.forward = utils_torch.parse.ResolveStr(param.Lambda, ObjCurrent=param.cache.__ResolveRef__)
        return
    def SetFullName(self, FullName):
        utils_torch.model.SetFullNameForModel(self, FullName)

__MainClass__ = LambdaLayer
utils_torch.model.SetMethodForModelClass(__MainClass__)