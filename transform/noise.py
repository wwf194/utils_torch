import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.attr import *
from utils_torch.module.AbstractModules import AbstractModule

class NoiseFromDistribution(utils_torch.module.AbstractModuleWithParam):
    # def __init__(self, param=None, data=None, **kw):
    #     super(NoiseGenerator, self).__init__()
    #     self.InitModule(self, param, data, ClassPath="utils_torch.transform.NoiseGenerator", **kw)
    HasTensor = False
    def __init__(self, **kw):
        super().__init__(**kw)
        return
    def BeforeBuild(self, IsLoad=False):
        super().BeforeBuild(IsLoad)
    def Build(self, IsLoad=False, **kw):
        super().BeforeBuild(IsLoad)
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
                elif param.Distribution in ["Laplacian"]:
                    # to be implemented
                    pass
                else:
                    raise Exception(param.Distribution)
            else:
                raise Exception(param.SubMethod)
        else:
            raise Exception(param.Method)
    def __call__(self, *Args, **Kw):
        return self.forward(*Args, **Kw)
class GaussianNoise(NoiseFromDistribution):
    def Build(self, IsLoad=False):
        super().BeforeBuild()
        param = self.param
        param.Distribution = "Gaussian"
        super().Build()