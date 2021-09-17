from attrs import EnsureAttrs
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.attrs import *

class SerialSender(nn.Module):
    def __init__(self, param=None):
        super(SerialSender, self).__init__()
        if param is not None:
            self.param = param
    def InitFromParam(self):
        param = self.param
        self.ContentList = []
        EnsureAttrs(param, "ExtractMethod.Initialize", default="Default")
        if GetAttrs(param.ExtractMethod.Initialize) in ["Default"]:
            self.ExtractMethod = self.DefaultExtractMethod
        elif param.ExtractMethod.Initialize in ["eval"]:
            self.ExtractMethod = eval(GetAttrs(param.ExtractMethod))
        else:
            raise Exception()
        return
    def Receive(self, content):
        self.ContentList = content
        self.NextSendIndex = 0
    def DefaultExtractMethod(self, Index):
        return self.ContentList[Index]
    def RegisterExtractMethod(self, method):
        self.ExtractMethod = method
    def Send(self):
        Content = self.ExtractMethod(Index=self.NextSendIndex)
        self.NextSendIndex += 1
        return Content