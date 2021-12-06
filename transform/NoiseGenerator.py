import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.attrs import *
from utils_torch.module.AbstractModules import AbstractModule
class NoiseGenerator(utils_torch.module.AbstractModuleWithParam):
    # def __init__(self, param=None, data=None, **kw):
    #     super(NoiseGenerator, self).__init__()
    #     self.InitModule(self, param, data, ClassPath="utils_torch.transform.NoiseGenerator", **kw)
    HasTensor = False
    def __init__(self, **kw):
        super().__init__(**kw)
        return
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
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
# utils_torch.transform.SetMethodForTransformModule(__MainClass__)