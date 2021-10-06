import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

class SingleLayer(nn.Module):
    def __init__(self):
        super(SingleLayer, self).__init__()
    def InitFromParam(self, param=None, IsLoad=False):
        if param is None:
            param = self.param
            data = self.data
            cache = self.cache
        else:
            self.param = param
            self.data = utils_torch.EmptyPyObj()
            self.cache = utils_torch.EmptyPyObj()

        cache.IsLoad = IsLoad
        cache.Tensors = []

        EnsureAttrs(param, "IsExciInhi", default=False)

        if not HasAttrs(param, "Output.Num") or not HasAttrs(param, "Input.Num"):
            if HasAttrs(param, "Weight.Size"):
                SetAttrs(param, "Input.Num", param.Weight.Size[0])
                SetAttrs(param, "Output.Num", param.Weight.Size[1])
            else:
                raise Exception()
    def SetBias(self):
        param = self.param
        data = self.data
        cache = self.cache        
        if GetAttrs(param.Bias):
            if cache.IsLoad:
                data.Bias = utils_torch.ToTorchTensor(data.Bias)
            else:
                data.Bias = (torch.zeros(param.Bias.Size, requires_grad=True))
            cache.Tensors.append([data, "Bias", data.Bias])
        else:
            data.Bias = 0.0
        self.GetBias = lambda:data.Bias
    def SetWeight(self):
        param = self.param
        data = self.data
        cache = self.cache
        EnsureAttrs(param, "Weight.Size", value=[param.Input.Num, param.Output.Num])
        EnsureAttrs(param, "Weight.Init", default=utils_torch.PyObj(
            {"Method":"kaiming", "Coefficient":1.0})
        )

        if self.cache.IsLoad:
            data.Weight = utils_torch.ToTorchTensor(data.Weight)
        else:
            data.Weight = utils_torch.model.CreateWeight2D(param.Weight)

        utils_torch.AddLog(
            str(utils_torch.PyObj({
                param.FullName + ".Weight": param.Weight
            })),
            logger="InitWeight",
            TimeStamp=False, File=False, LineNum=False,
        )
        utils_torch.AddLog(
            utils_torch.Tensor2Str(data.Weight),
            logger="InitWeight",
            TimeStamp=False, File=False, LineNum=False,
        )

        cache.Tensors.append([data, "Weight", data.Weight])
        GetWeightFunction = [lambda :data.Weight]

        EnsureAttrs(param.Weight, "NoSelfConnection", default=False)
        if param.IsExciInhi:
            param.Weight.IsExciInhi = param.IsExciInhi
            #utils_torch.model.ParseExciInhiNum(param.Weight)
            if not HasAttrs(param, "Weight.Excitatory.Num"):
                EnsureAttrs(param, "Weight.Excitatory.Ratio", default=0.8)
                SetAttrs(param, "Weight.Excitatory.Num", value=round(param.Weight.Excitatory.Ratio * param.Weight.Size[0]))
                SetAttrs(param, "Weight.Inhibitory.Num", value=param.Weight.Size[0] - param.Weight.Excitatory.Num)

            EnsureAttrs(param.Weight, "ConstraintMethod", value="AbsoluteValue")
            cache.WeightConstraintMethod = utils_torch.model.GetConstraintFunction(param.Weight.ConstraintMethod)
            GetWeightFunction.append(cache.WeightConstraintMethod)
            ExciInhiMask = utils_torch.model.CreateExcitatoryInhibitoryMask(*param.Weight.Size, param.Weight.Excitatory.Num, param.Weight.Inhibitory.Num)
            cache.ExciInhiMask = utils_torch.NpArray2Tensor(ExciInhiMask)
            cache.Tensors.append([cache, "ExciInhiMask", cache.ExciInhiMask])
            GetWeightFunction.append(lambda Weight:Weight * cache.ExciInhiMask)
        if GetAttrs(param.Weight, "NoSelfConnection"):
            if param.Weight.Size[0] != param.Weight.Size[1]:
                raise Exception("NoSelfConnection requires Weight to be square matrix.")
            SelfConnectionMask = utils_torch.model.CreateSelfConnectionMask(param.Weight.Size[0])
            cache.SelfConnectionMask = utils_torch.NpArray2Tensor(SelfConnectionMask)
            cache.Tensors.append([cache, "SelfConnectionMask", cache.SelfConnectionMask])
            GetWeightFunction.append(lambda Weight:Weight * self.SelfConnectionMask)
        self.GetWeight = utils_torch.StackFunction(GetWeightFunction, InputNum=0)
        return
    def GetTensorLocation(self):
        return self.cache.TensorLocation
    def GetTrainWeight(self):
        return self.cache.TrainWeight
    def SetTrainWeight(self):
        data = self.data
        cache = self.cache
        self.ClearTrainWeight()
        cache.TrainWeight = {}
        if hasattr(data, "Weight") and isinstance(data.Weight, torch.Tensor):
            cache.TrainWeight["Weight"] = data.Weight
        if hasattr(data, "Bias") and isinstance(data.Bias, torch.Tensor):
            cache.TrainWeight["Bias"] = data.Bias
        return cache.TrainWeight
    def ClearTrainWeight(self):
        cache = self.cache
        if hasattr(cache, "TrainWeight"):
            delattr(cache, "TrainWeight")
    # def SetLogger(self, logger):
    #     return utils_torch.model.SetLoggerForModel(self, logger)
    # def GetLogger(self):
    #     return utils_torch.model.GetLoggerForModel(self)
    # def Log(self, data, Name="Undefined"):
    #     return utils_torch.model.LogForModel(self, data, Name)
    def SetFullName(self, FullName):
        utils_torch.model.SetFullNameForModel(self, FullName)
    def PlotSelfWeight(self, SaveDir=None):
        param = self.param
        if SaveDir is None:
            SaveDir = utils_torch.GetSaveDir() + "weights/"
        
        if hasattr(param, "FullName"):
            FullName = param.FullName + "."
        else:
            FullName = ""

        Name = FullName + "Weight"
        utils_torch.plot.PlotWeightAndDistribution(
            weight=self.GetWeight(), Name=Name, SavePath=SaveDir + Name + ".svg"
        )
        if hasattr(self, "GetBias") and isinstance(self.GetBias(), torch.Tensor):
            Name = FullName + "Bias"
            utils_torch.plot.PlotWeightAndDistribution(
                weight=self.GetBias(), Name=Name, SavePath=SaveDir + Name + ".svg"
            )
    def SetPlotWeight(self, UseFullName=False):
        cache = self.cache
        param = self.param

        self.ClearPlotWeight()

        if UseFullName and hasattr(param, "FullName"):
            Name = param.FullName + "."
        else:
            Name = ""
        
        cache.PlotWeight = {
            Name + "Weight": self.GetWeight
        }

        if hasattr(self, "GetBias") and isinstance(self.GetBias(), torch.Tensor):
            cache.PlotWeight[Name + "Bias"] = self.GetBias

        return cache.PlotWeight
__MainClass__ = SingleLayer
utils_torch.model.SetMethodForModelClass(__MainClass__)