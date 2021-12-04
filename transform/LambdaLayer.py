import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.attrs import *

from utils_torch.module.AbstractModules import AbstractModule
class LambdaLayer(AbstractModule):
    def __init__(self, param=None, data=None, **kw):
        super(LambdaLayer, self).__init__()
        utils_torch.module.InitForModule(self, param, data, ClassPath="utils_torch.module.LambdaLayer", **kw)
    def InitFromParam(self, IsLoad=False):
        utils_torch.module.InitFromParamForModule(self, IsLoad)
        param = self.param
        self.forward = utils_torch.parse.ResolveStr(param.Lambda, ObjCurrent=param.cache.__ResolveRef__)
        return
    def SetFullName(self, FullName):
        utils_torch.module.SetFullNameForModule(self, FullName)

__MainClass__ = LambdaLayer
#utils_torch.module.SetMethodForModuleClass(__MainClass__)