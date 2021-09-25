import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.attrs import *

class LambdaLayer(nn.Module):
    def __init__(self, param=None):
        super(LambdaLayer, self).__init__()
        if param is not None:
            self.param = param
    def InitFromParam(self):
        param = self.param
        self.forward = utils_torch.parse.Resolve(param.Lambda, ObjCurrent=param.cache.__ResolveRef__)
        return

__MainClass__ = LambdaLayer