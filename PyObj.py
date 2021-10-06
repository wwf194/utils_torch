import utils_torch
from utils_torch.attrs import *

def EmptyPyObj():
    return PyObj()

def CheckIsPyObj(Obj):
    if not IsPyObj(Obj):
        raise Exception("Obj is not PyObj, but %s"%type(Obj))

def IsPyObj(Obj):
    return isinstance(Obj, PyObj)

def IsDictLikePyObj(Obj):
    return isinstance(Obj, PyObj) and not Obj.IsListLike()

def IsListLikePyObj(Obj):
    return isinstance(Obj, PyObj) and Obj.IsListLike()

class PyObjCache(object):
    def __init__(self):
        return

class PyObj(object):
    def __init__(self, param=None):
        self.cache = PyObjCache()
        if param is not None:
            if type(param) is dict:
                self.FromDict(param)
            elif type(param) is list:
                self.FromList(param)
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
    def __str__(self):
        return utils_torch.json.PyObj2JsonStr(self)
    def FromList(self, List):
        ListParsed = []
        for Index, Item in enumerate(List):
            if type(Item) is dict:
                ListParsed.append(PyObj(Item))
            elif type(Item) is list:
                ListParsed.append(PyObj(Item))
            else:
                ListParsed.append(Item)
        self.__value__ = ListParsed
        return self
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
                        valueOld = getattr(obj, key) 
                        if isinstance(valueOld, PyObj) and valueOld.IsDictLike():
                            if isinstance(value, PyObj) and value.IsDictLike():
                                valueOld.FromPyObj(value)
                            elif isinstance(value, dict):
                                valueOld.FromDict(value)
                            else:
                                # if hasattr(value, "__value__"):
                                #     utils_torch.AddWarning("PyObj: Overwriting key: %s. Original Value: %s, New Value: %s"\
                                #         %(key, getattr(obj, key), value))                                       
                                setattr(valueOld, "__value__", value)
                        else:
                            # utils_torch.AddWarning("PyObj: Overwriting key: %s. Original Value: %s, New Value: %s"\
                            #     %(key, valueOld, value))
                            setattr(obj, key, self.ProcessValue(key, value))
                    else:
                        if isinstance(obj, PyObj):
                            setattr(obj, key, self.ProcessValue(key, value))
                        else:
                            setattr(parent, parentAttr, PyObj({
                                "__value__": obj,
                                key: self.ProcessValue(key, value)
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
        return self
    def FromPyObj(self, Obj):
        self.FromDict(Obj.__dict__)
    def ProcessValue(self, key, value):
        if isinstance(value, dict):
            return PyObj(value)
        elif isinstance(value, list):
            if key in ["__value__"]:
                return value
            else:
                return PyObj(value)
        elif type(value) is PyObj:
            return value
        else:
            return value
    def ToList(self):
        if not self.IsListLike():
            raise Exception()
        return self.__value__
    def IsListLike(self):
        return hasattr(self, "__value__") and isinstance(self.__value__, list)
    def IsDictLike(self):
        return not self.IsListLike()
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
        Dict = {}
        for key, value in ListAttrsAndValues(self, Exceptions=["__ResolveRef__"]):
            if type(value) is PyObj:
                value = value.ToDict()
            Dict[key] = value
        return Dict