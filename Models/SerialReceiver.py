import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import utils_torch
from utils_torch.attrs import *

class SerialReceiver():
    def __init__(self, param=None, data=None):
        utils_torch.model.InitForModel(self, param, data, ClassPath="utils_torch.Models.SerialReceiver")
    def InitFromParam(self, IsLoad=False):
        cache = self.cache
        cache.IsLoad = IsLoad
        cache.IsInit = not IsLoad
        self.ContentList = []
        self.SetSendMethod()
        self.SetReceiveMethod()
        return
    def SetSendMethod(self):
        param = self.param
        cache = self.cache
        if cache.IsInit:
            EnsureAttrs(param, "Send.Method", default="Default")
        method = GetAttrs(param.Send.Method)
        if method in ["Default"]:
            self._Send = self._SendDefault
        elif method in ["Lambda", "eval"]:
            self._Send = eval(GetAttrs(param.Send.Args))
        else:
            raise Exception(method)
        return
    def SetReceiveMethod(self):
        param = self.param
        cache = self.cache
        if cache.IsInit:
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
        self.ContentList.append(content)
    def _SendDefault(self, List):
        return List
    def Send(self):
        result = self._Send(self.ContentList)
        self.ContentList = []
        return result
    def SendWithoutFlush(self):
        return self.ContentList

__MainClass__ = SerialReceiver
utils_torch.model.SetMethodForModelClass(__MainClass__)