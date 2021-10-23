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
utils_torch.model.SetMethodForModelClass(SignalHolder, HasTensor=False)

class SerialSender():
    def __init__(self, param=None, data=None, **kw):
        #super(SerialSender, self).__init__()
        utils_torch.model.InitForModel(self, param, data,  ClassPath="utils_torch.Modules.SerialSender", **kw)
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
utils_torch.model.SetMethodForModelClass(SerialSender, HasTensor=False)

class SerialReceiver():
    def __init__(self, param=None, data=None, **kw):
        utils_torch.model.InitForModel(self, param, data, ClassPath="utils_torch.Modules.SerialReceiver", **kw)
    def InitFromParam(self, IsLoad=False):
        utils_torch.model.InitFromParamForModel(self, IsLoad)
        cache = self.cache
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
utils_torch.model.SetMethodForModelClass(SerialReceiver, HasTensor=False)