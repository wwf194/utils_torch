import utils_torch
from utils_torch.attrs import *

def EmptyPyObj():
    return PyObj()

def CheckIsPyObj(Obj):
    if not IsPyObj(Obj):
        raise Exception("Obj is not PyObj, but %s"%type(Obj))

def IsPyObj(Obj):
    return isinstance(Obj, PyObj)

def IsPyObjAndDictLike(Obj):
    return isinstance(Obj, PyObj) and not Obj.IsListLike()

def IsPyObjAndListLike(Obj):
    return isinstance(Obj, PyObj) and Obj.IsListLike()

class PyObjCache(object):
    def __init__(self):
        return

class PyObj(object):
    def __init__(self, param=None):
        # if param in ["cache"]:
        #     self.IsCache = True
        #     return
        # else:
        #     self.IsCache = False
        self.cache = PyObjCache()
        if param is not None:
            if type(param) is dict:
                self.FromDict(param)
            elif type(param) is list:
                self.FromList(param)
            elif type(param) is str:
                param = json.loads(param)
            else:
                raise Exception()
    def __repr__(self):
        return str(self.ToDict())
    def __setitem__(self, key, value):
        if hasattr(self, "__value__") and isinstance(self.__value__, list):
            self.__value__[key] = value
        else:
            self.__dict__[key] = value
    def __getitem__(self, index):
        if hasattr(self, "__value__") and isinstance(self.__value__, list):
            return self.__value__[index]
        else:
            return self.__dict__[index]
    def __len__(self):
        return len(self.__value__)
    
    def FromList(self, list):
        ListParsed = []
        for Index, Item in enumerate(list):
            if type(Item) is dict:
                ListParsed.append(PyObj(Item))
            elif type(Item) is list:
                ListParsed.append(self.FromList(Item))
            else:
                ListParsed.append(Item)
        return ListParsed
    def FromDict(self, Dict):
        #self.__dict__ = {}
        for key, value in Dict.items():
            # if key in ["Init.Method"]:
            #     print("aaa")

            if key in ["__ResolveRef__"]:
                if not hasattr(self, "__ResolveRef__"):
                    setattr(self.cache, key, value)
                continue

            if "." in key:
                keys = key.split(".")
            else:
                keys = [key]
            utils_torch.python.CheckIsLegalPyName(key[0])
            obj = self
            parent, parentAttr = None, None
            for index, key in enumerate(keys):
                if index == len(keys) - 1:
                    if hasattr(obj, key):
                        if isinstance(getattr(obj, key), PyObj):
                            if isinstance(value, PyObj):
                                getattr(obj, key).FromPyObj(value)
                            elif isinstance(value, dict):
                                getattr(obj, key).FromDict(value)
                            else:
                                if hasattr(getattr(obj, key), "__value__"):
                                    utils_torch.AddWarning("PyObj: Overwriting key: %s. Original Value: %s, New Value: %s"\
                                        %(key, getattr(obj, key), value))                                       
                                setattr(getattr(obj, key), "__value__", value)
                        else:
                            utils_torch.AddWarning("PyObj: Overwriting key: %s. Original Value: %s, New Value: %s"\
                                %(key, getattr(obj, key), value))
                            setattr(obj, key, self.ProcessValue(value))
                    else:
                        if isinstance(obj, PyObj):
                            setattr(obj, key, self.ProcessValue(value))
                        else:
                            setattr(parent, parentAttr, PyObj({
                                "__value__": obj,
                                key: self.ProcessValue(value)
                            }))
                else:
                    if hasattr(obj, key):
                        parent, parentAttr = obj, key
                        obj = getattr(obj, key)
                    else:
                        if isinstance(obj, PyObj):
                            setattr(obj, key, PyObj())
                            parent, parentAttr = obj, key
                            obj = getattr(obj, key)
                        else:
                            setattr(parent, parentAttr, PyObj({
                                "__value__": obj,
                                key: PyObj()
                            }))
                            obj = getattr(parent, parentAttr)
                            parent, parentAttr = obj, key
                            obj = getattr(obj, key)
            # else:
            #     CheckIsLegalPyName(key)
            #     value = self.ProcessValue(value)
            #     if isinstance(value, PyObj):
            #         self.FromPyObj(value)
            #     else:

            #     setattr(self, key, self.ProcessValue(value))
            # self.SetAttr(self, key, value)
        return self
    def FromPyObj(self, Obj):
        self.FromDict(Obj.__dict__)
    def ProcessValue(self, value):
        if isinstance(value, dict):
            return PyObj(value)
        elif isinstance(value, list):
            return PyObj().FromList(value)
        elif type(value) is PyObj:
            return value
        else:
            return value
    def IsListLike(self):
        return hasattr(self, "__value__") and isinstance(self.__value__, list)
    def IsDictLike(self):
        return not self.IsDictLike()
    def SetResolveBase(self):
        self.__ResolveBase__ = True
    def IsResolveBase(self):
        if hasattr(self, "__ResolveBase__"):
            if self.__ResolveBase__==True or self.__ResolveBase__ in ["here"]:
                return True
        return False
    def append(self, content):
        if not self.IsListLike():
            raise Exception()
        self.__value__.append(content)
    def ToDict(self):
        d = {}
        for key, value in ListAttrsAndValues(self, Exceptions=["__ResolveRef__"]):
            if type(value) is PyObj:
                value = value.ToDict()
            d[key] = value
        return d