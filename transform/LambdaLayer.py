import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.attrs import *

from utils_torch.module.AbstractModules import AbstractModule
class LambdaLayer(AbstractModule):
    def __init__(self, param=None, data=None, **kw):
        super(LambdaLayer, self).__init__()
        utils_torch.transform.InitForModule(self, param, data, ClassPath="utils_torch.transform.LambdaLayer", **kw)
    def InitFromParam(self, IsLoad=False):
        utils_torch.transform.InitFromParamForModule(self, IsLoad)
        param = self.param
        self.forward = utils_torch.parse.ResolveStr(param.Lambda, ObjCurrent=param.cache.__ResolveRef__)
        return
    def SetFullName(self, FullName):
        utils_torch.transform.SetFullNameForModule(self, FullName)

__MainClass__ = LambdaLayer
#utils_torch.transform.SetMethodForTransformModule(__MainClass__)