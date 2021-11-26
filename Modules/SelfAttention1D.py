import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

class SelfAttention1D(nn.Module):
    def __init__(self, param=None, data=None, **kw):
        super(SelfAttention1D, self).__init__()
        utils_torch.module.InitForModule(self, param, data, ClassPath="utils_torch.Modules.Bias", **kw)

    def InitFromParam(self, IsLoad=False):
        utils_torch.module.InitFromParamForModule(self, IsLoad)
        param = self.param
        data = self.data
        cache = self.cache
        
        assert HasAttrs(param, "Input.Num")
        assert HasAttrs(param, "Output.Num")
        EnsureAttrs(param, "Attention.Feature.Num", default=param.Input.Num)
        SetAttrs(param, "Weight.Input2Query.Size", value=[param.Input.Num, param.Attention.Feature.Num])
        SetAttrs(param, "Weight.Input2Key.Size",   value=[param.Input.Num, param.Attention.Feature.Num])
        if HasAttrs(param, "Output.Num"):
            SetAttrs(param, "Weight.Input2Value.Size", value=[param.Input.Num, param.Output.Num])
        elif HasAttrs(param, "Weight.Input2Value.Size"):
            SetAttrs(param, "Output.Num", value=param.Weight.Input2Value.Size[1])
        else:
            raise Exception()

        if cache.IsInit:
            data.Input2Query = utils_torch.module.CreateWeight2D(param.Weight.Input2Query)
            data.Input2Key = utils_torch.module.CreateWeight2D(param.Weight.Input2Key)
            data.Input2Value = utils_torch.module.CreateWeight2D(param.Weight.Input2Value)
        else:
            data.Input2Query = utils_torch.ToTorchTensor(data.Input2Query)
            data.Input2Key   = utils_torch.ToTorchTensor(data.Input2Key)
            data.Input2Value = utils_torch.ToTorchTensor(data.Input2Value)

        cache.Tensors.append([data, "Input2Query", data.Input2Query])
        cache.Tensors.append([data, "Input2Key",   data.Input2Key])
        cache.Tensors.append([data, "Input2Value", data.Input2Value])
        
        cache.AttentionCoefficient = 1.0 / param.Attention.Feature.Num ** 0.5
    def SplitHead(self, Data, HeadNum):
        BatchSize = Data.shape[0]
        TokenNum = Data.shape[1]
        # FeatureNumTotal = Data.shape[2]
        Data = Data.view(BatchSize, TokenNum, HeadNum, -1)
        Data = Data.permute(0, 2, 1, 3)
        return Data
    def forward(self, Input, log):
        # Input: [BatchSize, TokenNum, InputNum]
        data = self.data
        cache = self.cache
        Query = torch.mm(Input, data.Input2Query) # [BatchSize, TokenNum, AttentionFeatureNum]
        Key = torch.mm(Input, data.Input2Key) # [BatchSize, TokenNum, AttentionFeatureNum]
        Value = torch.mm(Input, data.Input2Value) # [BatchSize, TokenNum, OutputNum]

        self.LogCache("Attention.Key", Key, "Attention", log=log)
        self.LogCache("Attention.Query", Query, "Attention", log=log)
        self.LogCache("Attention.Value", Value, "Attention", log=log)

        if cache.MultiHead:
            QueryMultiHead = torch.mm(Query, data.Query2MultiHeadQuery) # [BatchSize, TokenNum, AttentionMultiHeadFeatureNum]
            KeyMultiHead   = torch.mm(Key,   data.Query2MultiHeadKey)   # [BatchSize, TokenNum, AttentionMultiHeadFeatureNum]
            ValueMultiHead = torch.mm(Value, data.Query2MultiHeadValue) # [BatchSize, TokenNum, OutputFeatureNum]

            QueryMultiHead = self.SplitHead(QueryMultiHead, cache.HeadNum) # [BatchSize, HeadNum, TokenNum, AttentionPerHeadFeatureNum]
            KeyMultiHead   = self.SplitHead(KeyMultiHead,   cache.HeadNum)
            ValueMultiHead = self.SplitHead(ValueMultiHead, cache.HeadNum)

            AttentionMultiHead = torch.einsum("bhtf,bhtf->bht", QueryMultiHead, KeyMultiHead) # [BatchSize, HeadNum, TokenNum, TokenNum]
            AttentionMultiHead *= cache.AttentionPerHeadFeatureNum
            AttentionMultiHead = F.softmax(AttentionMultiHead, axis=3)

            OutputMultiHead = torch.einsum("bhtu,bhuo->bhto", AttentionMultiHead, ValueMultiHead)
            
            self.LogCache("AttentionMultiHead",       AttentionMultiHead, "Attention", log=log)
            self.LogCache("AttentionMultiHead.Query", QueryMultiHead,     "Attention", log=log)
            self.LogCache("AttentionMultiHead.Key",   KeyMultiHead,       "Attention", log=log)
            self.LogCache("AttentionMultiHead.Value", ValueMultiHead,     "Attention", log=log)
            self.LogCache("OutputMultiHead",          OutputMultiHead,    "Activity",  log=log)
            return OutputMultiHead
        else:
            Attention = torch.bmm(Query, Key.permute(0, 2, 1)) ** cache.AttentionFeatureNum # [BatchSize, TokenNum, TokenNum]
            Attention = F.softmax(Attention, dim=2) # [BatchSize, TokenNum, TokenNum]
            Output = torch.bmm(Attention, Value) # [BatchSize, TokenNum, OutputNum]

            self.LogCache("Attention", Attention, "Attention", log=log)
            self.LogCache("Output", Output, "Activity", log=log)
            return Output
__MainClass__ = SelfAttention1D
utils_torch.module.SetMethodForModuleClass(__MainClass__)