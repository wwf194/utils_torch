import utils_torch
from utils_torch.attrs import *
class SignalHolder():
    def __init__(self, param=None, data=None, **kw):
        kw.setdefault("HasTensor", False)
        utils_torch.model.InitForModel(self, param, data, ClassPath="utils_torch.Modules.SignalHolder", **kw)
    def InitFromParam(self, IsLoad=False):
        utils_torch.model.InitFromParamForModel(self, IsLoad)
    def Receive(self, Obj):
        self.cache.Content = Obj
    def Send(self):
        return self.cache.Content
    def Clear(self):
        utils_torch.attrs.RemoveAttrIfExists(self.cache, "Content")