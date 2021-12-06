import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.attrs import *

from utils_torch.transform import AbstractTransform
class LambdaLayer(AbstractTransform):
    # def __init__(self, param=None, data=None, **kw):
    #     super(LambdaLayer, self).__init__()
    #     #self.InitModule(self, param, data, ClassPath="utils_torch.transform.LambdaLayer", **kw)
    def __init__(self, **kw):
        super().__init__(**kw)
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        param = self.param
        self.forward = utils_torch.parse.ResolveStr(param.Lambda, ObjCurrent=param.cache.__ResolveRef__)
        return
    def SetFullName(self, FullName):
        utils_torch.transform.SetFullNameForModule(self, FullName)

__MainClass__ = LambdaLayer
#utils_torch.transform.SetMethodForTransformModule(__MainClass__)