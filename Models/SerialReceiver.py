import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SerialReceiver(nn.Module):
    def __init__(self, param=None):
        super(SerialReceiver, self).__init__()
        if param is not None:
            self.param = param
    def InitFromParam(self):
        self.ContentList = []
        return
    def Receive(self, content):
        self.ContentList.append()
    def Send(self):
        ContentList = self.ContentList
        self.ContentList = []
        return ContentList
    def SendWithoutFlush(self):
        return self.ContentList

__MainClass__ = SerialReceiver