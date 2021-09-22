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
        EnsureAttrs(param, "ExtractMethod.Init.Method", default="Default")
        method = GetAttrs(param.ExtractMethod.Init.Method)
        if method in ["Default"]:
            self.ExtractMethod = self.DefaultExtractMethod
        elif method in ["eval"]:
            ExtractMethodStr = GetAttrs(param.ExtractMethod.Init.Args)
            SetAttrs(param, "ExtractMethod.Description", value=ExtractMethodStr)
            self.ExtractMethod = eval(ExtractMethodStr)
        else:
            raise Exception()
        return
    def Receive(self, content):
        self.ContentList = content
        self.NextSendIndex = 0
    def DefaultExtractMethod(self, ContentList, Index):
        return ContentList[Index]
    def RegisterExtractMethod(self, method):
        self.ExtractMethod = method
    def Send(self):
        Content = self.ExtractMethod(self.ContentList, Index=self.NextSendIndex)
        self.NextSendIndex += 1
        return Content

__MainClass__ = SerialSender