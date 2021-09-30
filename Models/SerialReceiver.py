import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import utils_torch
from utils_torch.attrs import *

class SerialReceiver(nn.Module):
    def __init__(self, param=None):
        super(SerialReceiver, self).__init__()
        if param is not None:
            self.param = param
    def InitFromParam(self):
        self.ContentList = []
        self.SetSendMethod()
        self.SetReceiveMethod()
        return
    def SetSendMethod(self):
        param = self.param
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
    def SetFullName(self, FullName):
        utils_torch.model.SetFullNameForModel(self, FullName)

__MainClass__ = SerialReceiver