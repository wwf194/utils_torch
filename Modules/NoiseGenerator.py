import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.attrs import *

class NoiseGenerator(nn.Module):
    def __init__(self, param=None, data=None, **kw):
        super(NoiseGenerator, self).__init__()
        utils_torch.module.InitForModule(self, param, data, ClassPath="utils_torch.Modules.NoiseGenerator", **kw)
    def InitFromParam(self, IsLoad=False):
        utils_torch.module.InitFromParamForModule(self, IsLoad)
        param = self.param
        cache = self.cache
        if param.Method in ["Adaptive"]:
            if param.SubMethod in ["FromInputStd"]:
                if param.Distribution in ["Gaussian"]:
                    self.forward = lambda Input: \
                        utils_torch.math.SampleFromGaussianDistributionTorch(
                            Mean=0.0,
                            Std=torch.std(Input.detach()).item() * param.StdRatio,
                            Shape=tuple(Input.size()),
                        ).to(self.GetTensorLocation())
                else:
                    raise Exception(param.Distribution)
            else:
                raise Exception(param.SubMethod)
        else:
            raise Exception(param.Method)
    def SetTensorLocation(self, Location):
        cache = self.cache
        cache.TensorLocation = Location
    def GetTensorLocation(self):
        return self.cache.TensorLocation

__MainClass__ = NoiseGenerator
utils_torch.module.SetMethodForModuleClass(__MainClass__)