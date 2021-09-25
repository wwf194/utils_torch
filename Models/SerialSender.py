import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils_torch.attrs import *

class SerialSender(nn.Module):
    def __init__(self, param=None):
        super(SerialSender, self).__init__()
        if param is not None:
            self.param = param
    def InitFromParam(self):
        param = self.param
        self.ContentList = []
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
        self.ContentList = content
        self.NextSendIndex = 0
    def _SendDefault(self, ContentList, Index):
        return ContentList[Index]
    def RegisterExtractMethod(self, method):
        self.ExtractMethod = method
    def Send(self):
        Content = self._Send(self.ContentList, Index=self.NextSendIndex)
        self.NextSendIndex += 1
        return Content

__MainClass__ = SerialSender