import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils_torch.attrs import *

class SerialSender():
    def __init__(self, param=None, data=None, **kw):
        #super(SerialSender, self).__init__()
        utils_torch.model.InitForModel(self, param, data,  ClassPath="utils_torch.Models.SerialSender", **kw)
    def InitFromParam(self, IsLoad=False):
        utils_torch.model.InitFromParamForModel(self, IsLoad)
        param = self.param
        cache = self.cache
        cache.ContentList = []
        self.SetSendMethod()
        self.SetReceiveMethod()
    def SetSendMethod(self):
        param = self.param
        EnsureAttrs(param, "Send.Method", default="Default")
        method = GetAttrs(param.Send.Method)
        if method in ["Default"]:
            self._Send = self.SendDefault
        elif method in ["Lambda", "eval"]:
            self._Send = eval(GetAttrs(param.Send.Args))
        else:
            raise Exception(method)
        return
    def SetReceiveMethod(self):
        param = self.param
        EnsureAttrs(param, "Receive.Method", default="Default")
        method = GetAttrs(param.Receive.Method)
        if method in ["Default"]:
            self.Receive = self.ReceiveDefault
        elif method in ["Lambda", "eval"]:
            self.Receive = eval(GetAttrs(param.Receive.Args))
        else:
            raise Exception(method)
        return
    def ReceiveDefault(self, content):
        cache = self.cache
        cache.ContentList = content
        cache.NextSendIndex = 0
    def _SendDefault(self, ContentList, Index):
        return ContentList[Index]
    def RegisterExtractMethod(self, method):
        self.ExtractMethod = method
    def Send(self):
        cache = self.cache
        Content = self._Send(cache.ContentList, Index=cache.NextSendIndex)
        cache.NextSendIndex += 1
        return Content

__MainClass__ = SerialSender
utils_torch.model.SetMethodForModelClass(__MainClass__)