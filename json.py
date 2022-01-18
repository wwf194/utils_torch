import re
import json
import json5
from typing import List

import utils_torch
from utils_torch.python import CheckIsLegalPyName
from utils_torch.attr import *
# from utils_torch.utils import ListAttrs # leads to recurrent reference.

import numpy as np

def JsonObj2PyObj(JsonObj):
    if isinstance(JsonObj, list):
        return utils_torch.EmptyPyObj().FromList(JsonObj)
    elif isinstance(JsonObj, dict):
        return utils_torch.EmptyPyObj().FromDict(JsonObj)
    else:
        raise Exception()
Dict2PyObj = JsonObj2PyObj

def JsonObj2JsonStr(JsonObj):
    return json.dumps(JsonObj, indent=4, sort_keys=False)

def PyObj2JsonObj(Obj):
    return _PyObj2JsonObj(Obj, AttrStr="root")

def _PyObj2JsonObj(Obj, **kw):
    AttrStr = kw["AttrStr"]
    if isinstance(Obj, utils_torch.PyObj) and Obj.IsListLike():
        if len(ListAttrsAndValues(Obj))==1:
            Obj = Obj.ToList()
    if isinstance(Obj, list) or isinstance(Obj, tuple):
        JsonObj = []
        for Index, Item in enumerate(Obj):
            kw["AttrStr"] = AttrStr + ".%d"%Index
            JsonObj.append(_PyObj2JsonObj(Item, **kw))
        return JsonObj
    elif isinstance(Obj, utils_torch.PyObj):
        JsonObj = {}
        for attr, value in ListAttrsAndValues(Obj, Exceptions=[]):
            if attr in ["__ResolveRef__"]:
                continue
            kw["AttrStr"] = AttrStr + ".%s"%attr
            JsonObj[attr] = _PyObj2JsonObj(value, **kw)
        return JsonObj
    elif isinstance(Obj, dict):
        JsonObj = {}
        for key, value in Obj.items():
            kw["AttrStr"] = AttrStr +  ".%s"%key
            JsonObj[key] = _PyObj2JsonObj(value, **kw)
        return JsonObj
    elif type(Obj) in [str, int, bool, float]:
        return Obj
    elif isinstance(Obj, np.ndarray):
        return utils_torch.NpArray2List(Obj)
    elif isinstance(Obj, np.float32) or isinstance(Obj, np.float64):
        return float(Obj)
    else:
        utils_torch.AddWarning("%s is of Type: %s Unserializable."%(kw["AttrStr"], type(Obj)))
        return "UnserializableObject of type: %s"%type(Obj)

def PyObj2DataObj(Obj):
    if isinstance(Obj, utils_torch.PyObj) and Obj.IsListLike():
        if len(ListAttrsAndValues(Obj))==1:
            Obj = Obj.ToList()
    if isinstance(Obj, list) or isinstance(Obj, tuple):
        JsonObj = []
        for Item in Obj:
            JsonObj.append(PyObj2DataObj(Item))
        return JsonObj
    elif isinstance(Obj, utils_torch.PyObj):
        JsonObj = {}
        for attr, value in ListAttrsAndValues(Obj, Exceptions=[]):
            if attr in ["__ResolveRef__"]:
                continue
            JsonObj[attr] = PyObj2DataObj(value)
        return JsonObj
    elif isinstance(Obj, dict):
        JsonObj = {}
        for key, value in Obj.items():
            JsonObj[key] = PyObj2DataObj(value)
        return JsonObj
    else:
        return Obj

def PyObj2JsonFile(Obj, FilePath):
    JsonStr = PyObj2JsonStr(Obj)
    JsonStr2JsonFile(JsonStr, FilePath)

def JsonStr2JsonFile(JsonStr, FilePath):    
    if FilePath.endswith(".jsonc") or FilePath.endswith(".json"):
        pass
    else:
        FilePath += ".jsnonc"
    utils_torch.EnsureFileDir(FilePath)
    with open(FilePath, "w") as f:
        f.write(JsonStr)

def JsonDumpsLambda(obj):
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return "UnserializableObject"

def PyObj2JsonStr(Obj):
    # return json.dumps(obj.__dict__, cls=change_type,indent=4)
    # why default=lambda o: o.__dict__?
    #return json.dumps(obj, default=lambda o: o.__dict__, sort_keys=False, indent=4)
    JsonObj = PyObj2JsonObj(Obj)
    return JsonObj2JsonStr(JsonObj)
    #return json.dumps(obj, default=JsonDumpsLambda, sort_keys=False, indent=4)

def PyObj2JsonStrHook(Obj):
    # to be implemented
    return

def JsonStr2JsonObj(JsonStr):
    RemoveJsonStrComments(JsonStr)
    JsonStr = re.sub(r"\s", "", JsonStr)
    _JsonStr2JsonObj(JsonStr)

def _JsonStr2JsonObj(JsonStr):
    #JsonStr = re.sub(r"\".*\"", "", JsonStr)
    #JsonStr = re.sub(r"\'.*\'", "", JsonStr)
    if JsonStr[0]=="{" and JsonStr[-1]=="}":
        Obj = {}
        Segments = JsonStr[1:-1].rstrip(",").split(",")
        for Segment in Segments:
            AttrValue = Segment.split(":")
            Attr = AttrValue[0]
            if not (Attr[0]=="\"" and Attr[-1]=="\"" or Attr[0]=="\'" and Attr[-1]=="\'"):
                raise Exception()
            Attr = Attr[1:-1]
            Value = AttrValue[1]
            Obj[Attr] = _JsonStr2JsonObj(Value)
        return Obj
    elif JsonStr[0]=="[" and JsonStr[-1]=="]":
        Obj = []
        Segments = JsonStr[1:-1].rstrip(",").split(",")
        for Index, Segment in enumerate(Segments):
            Obj.append(_JsonStr2JsonObj(Segment, ))    
        return Obj
    elif JsonStr[0]=="\"" and JsonStr[-1]=="\"" or JsonStr[0]=="\'" and JsonStr[-1]=="\'":
        return JsonStr[1:-1]
    try:
        Obj = int(JsonStr)
        return Obj
    except Exception:
        pass

    try:
        Obj = float(JsonStr)
        return Obj
    except Exception:
        pass
    raise Exception()

def JsonStr2PyObj(JsonStr):
    JsonObj = JsonStr2JsonObj(JsonStr)
    return JsonObj2PyObj(JsonObj)
    # return json.loads(JsonStr, object_hook=lambda d: SimpleNamespace(**d))
JsonStr_to_object = JsonStr2PyObj

def JsonFile2PyObj(FilePath):
    JsonObj = JsonFile2JsonObj(FilePath)
    Obj = JsonObj2PyObj(JsonObj)
    return Obj

def JsonFile2JsonObj(FilePath):
    # with open(FilePath, "r") as f:
    #     JsonStrLines = f.readlines()
    # JsonStrLines = RemoveJsonStrLinesComments(JsonStrLines)
    # JsonStr = "".join(JsonStrLines)
    # JsonStr = re.sub("\s", "", JsonStr) # Remove All Empty Characters
    # JsonStr = RemoveJsonStrComments(JsonStr)
    # return JsonStr2JsonObj(JsonStr)
    with open(FilePath, "r") as f:
        JsonObj = json5.load(f) # json5 allows comments
    return JsonObj

def RemoveJsonStrLinesComments(JsonStrLines): # Remove Single Line Comments Like //...
    for Index, JsonStrLine in enumerate(JsonStrLines):
        JsonStrLines[Index] = re.sub(r"//.*\n", "", JsonStrLine)
    return JsonStrLines

def RemoveJsonStrComments(JsonStr):
    JsonStr = re.sub(r"/\*.*\*/", "", JsonStr)
    return JsonStr

def JsonObj2JsonFile(JsonObj, FilePath):
    return JsonStr2JsonFile(JsonObj2JsonStr(JsonObj), FilePath)

def JsonStr2JsonFile(JsonStr, FilePath):
    return utils_torch.Str2File(JsonStr, FilePath)

def IsJsonObj(Obj):
    return \
    isinstance(Obj, utils_torch.PyObj) or \
    isinstance(Obj, str) or \
    isinstance(Obj, bool) or \
    isinstance(Obj, int) or \
    isinstance(Obj, float) or \
    isinstance(Obj, list) or \
    isinstance(Obj, dict) or \
    isinstance(Obj, tuple)

import pickle
def JsonObj2DataFile(Obj, FilePath):
    utils_torch.EnsureFileDir(FilePath)
    with open(FilePath, "wb") as f:
        pickle.dump(Obj, f)

def DataFile2JsonObj(FilePath):
    with open(FilePath, "rb") as f:
        JsonObj = pickle.load(f)
    return JsonObj

def DataFile2PyObj(FilePath):
    DataObj = DataFile2JsonObj(FilePath)
    return JsonObj2PyObj(DataObj)

def PyObj2DataFile(PyObj, FilePath):
    DataObj = PyObj2DataObj(PyObj)
    JsonObj2DataFile(DataObj, FilePath)

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
    # Class that represents a json-like data structure, recursively.
    # All instances are neither dict-like or list-like, resembling dicts or lists in json files.
    def __init__(self, param=None, data=None, **kw):
        self.cache = PyObjCache()
        if param is not None:
            if type(param) is dict:
                self.FromDict(param)
            elif type(param) is list:
                self.FromList(param)
            elif type(param) is PyObj:
                self.FromPyObj(param)
            else:
                raise Exception(type(param))
    def __hasattr__(self, Name):
        HasAttr = Name in self.__dict__ or Name=="__dict__"
        return HasAttr
    def __getattr__(self, Name):
        Obj = PyObj().SetCreateFromGetAttr(True).SetParentAndAttr(self, Name)
        setattr(self, Name, Obj)
        return Obj
    def SetParentAndAttr(self, parent, attr):
        self.cache.parent = parent
        self.cache.parentAttr = attr
        return self
    def HasParent(self):
        return "parent" in self.cache.__dict__ and "parentAttr" in self.cache.__dict__
    def GetParent(self):
        return self.cache.parent
    def GetParentAttr(self):
        return self.cache.parentAttr
    def SetDefault(self, Value):
        assert hasattr(self.cache, "parent")
        if self.IsCreateFromGetAttr() and self.IsEmpty():
            self.ResetParentAttr(Value)
        return self
    def SetValue(self, Value):
        self.__value__ = Value
        if len(self.__dict__)==2 and self.HasParent(): #['cache', '__value__']
            self.ResetParentAttr(Value)
        return self
    def HasValue(self):
        return "__value__" in self.__dict__
    def ResetParentAttr(self, Value):
        self.GetParent().SetAttr(self.GetParentAttr(), Value)
    def SetAttr(self, Name, Value):
        setattr(self, Name, Value)
        return getattr(self, Name)
    def RemoveAttr(self, Attr):
        self.__dict__.pop(Attr)
    def EnsureAttrs(self, AttrList, DefaultValue):
        if len(AttrList)==1:
            Attr = AttrList[0]
            if hasattr(self, Attr):
                Child = getattr(self, Attr)
                if isinstance(Child, PyObj):
                    # 2 special cases
                    if isinstance(DefaultValue, dict):
                        Child.FromDict(DefaultValue)
                        return
                    elif isinstance(DefaultValue, PyObj):
                        Child.FromPyObj(DefaultValue)
                        return

                    if Child.HasValue():
                        pass
                    else:
                        Child.SetValue(DefaultValue)
                else:
                    #setattr(self, Attr, DefaultValue)
                    pass
            else:
                setattr(self, Attr, DefaultValue)
        elif len(AttrList)>1:
            Child = getattr(self, AttrList[0])
            if not isinstance(Child, PyObj):
                setattr(self, AttrList[0], PyObj().SetValue(Child))
            Child = getattr(self, AttrList[0])
            Child.EnsureAttrs(AttrList[1:], DefaultValue)
        else:
            raise Exception(len(AttrList))
    def SetAttrs(self, AttrList, DefaultValue):
        if len(AttrList)==1:
            Attr = AttrList[0]
            if hasattr(self, Attr):
                Child = getattr(self, Attr)
                if isinstance(Child, PyObj):
                    # 2 special cases
                    if isinstance(DefaultValue, dict):
                        Child.FromDict(DefaultValue)
                        return
                    elif isinstance(DefaultValue, PyObj):
                        Child.FromPyObj(DefaultValue)
                        return

                    Child.SetValue(DefaultValue)
                else:
                    setattr(self, Attr, DefaultValue)
            else:
                setattr(self, Attr, DefaultValue)
        elif len(AttrList)>1:
            Child = getattr(self, AttrList[0])
            if not isinstance(Child, PyObj):
                setattr(self, AttrList[0], PyObj().SetValue(Child))
            Child = getattr(self, AttrList[0])
            Child.SetAttrs(AttrList[1:], DefaultValue)
        else:
            raise Exception(len(AttrList))
    def SetCreateFromGetAttr(self, value=True):
        self.cache.CreateFromGetAttr = value
        return self
    def IsCreateFromGetAttr(self):
        if hasattr(self.cache, "CreateFromGetAttr") and self.cache.CreateFromGetAttr is True:
            return True
        else:
            return False
    def __repr__(self):
        return str(self.ToDict())
    def __setitem__(self, key, value):
        if self.IsListLike():
            self.__value__[key] = value
        else:
            self.__dict__[key] = value
    def __contains__(self, item): # to support in operator
        if self.IsListLike():
            return item in self.__value
        else:
            return item in self.__dict__
    def __getitem__(self, index):
        if self.IsListLike():
            return self.__value__[index]
        else:
            return self.__dict__[index]
    def __len__(self):
        if self.IsListLike():
            return len(self.__value__)
        else:
            return len(self.__dict__) - 1
    def __str__(self):
        return utils_torch.json.PyObj2JsonStr(self)
    def setdefault(self, attr, value):
        if attr in self.__dict__:
            return getattr(self, attr)
        else:
            setattr(self, attr, value)
            return value
    def FromList(self, List, InPlace=True):
        ListParsed = []
        for Index, Item in enumerate(List):
            if type(Item) is dict:
                ListParsed.append(PyObj(Item))
            elif type(Item) is list:
                ListParsed.append(PyObj(Item))
            else:
                ListParsed.append(Item)
        if InPlace:
            self.__value__ = ListParsed
            return self
        else:
            return ListParsed
    def __add__(self, PyObj2):
        PyObj1 = self
        if PyObj1.IsListLike() and PyObj2.IsListLike():
            return PyObj1.__value__ + PyObj2.__value__
        else: # To Be Implemented: suuport for DictLikePyObj
            raise Exception()
    def __call__(self, *Args, **kw):
        return utils_torch.functions.CallGraph(
            self, *Args, **kw
        )
    def Update(self, Dict):
        self.FromDict(Dict)
        return self
    def FromDict(self, Dict):
        # Dict keys in form of "A.B.C" are supported.
        # {"A.B": "C"} will be understood as {"A": {"B": "C"}}
        for key, value in Dict.items():
            # if key in ["__ResolveRef__"]:
            #     if not hasattr(self, "__ResolveRef__"):
            #         setattr(self.cache, key, value)
            #     continue
            if "." in key: # and "|-->" not in key:
                keys = key.split(".")
            else:
                keys = [key]
            #utils_torch.python.CheckIsLegalPyName(keys[0])
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
    def Copy(self):
        # to be implemented: also copy cache
        return utils_torch.PyObj(self.__dict__)
    def FromPyObj(self, Obj):
        # to be implemented: also copy cache
        self.FromDict(Obj.__dict__)
        return self
    def ProcessValue(self, key, value):
        if isinstance(value, dict):
            return PyObj(value)
        elif isinstance(value, list):
            if key in ["__value__"]:
                return self.FromList(value, InPlace=False)
            else:
                return PyObj(value)
        elif type(value) is PyObj:
            return value
        else:
            return value
    def IsEmpty(self):
        return len(self.__dict__)==1
    def IsDictLike(self):
        return not self.IsListLike()
    def IsListLike(self):
        return hasattr(self, "__value__") and (
            isinstance(self.__value__, list) or isinstance(self.__value__, np.ndarray)
        )
    def ToDict(self):
        assert self.IsDictLike(), "Only DictLike PyObj supports ToDict()"
        Dict = dict(self.__dict__)
        Dict.pop("cache")
        return Dict
    def Items(self, Exceptions=["__IsResolveBase__"]):
        assert self.IsDictLike(), "Only DictLike PyObj supports items()"
        return ListAttrsAndValues(self, Exceptions=Exceptions)
    def ToList(self):
        assert self.IsListLike(), "Only ListLike PyObj supports ToList()"
        return self.__value__
    def RemoveAttrsIfExist(self, *Attrs):
        for Attr in Attrs:
            if hasattr(self, Attr):
                delattr(self, Attr)
        return self
    def Append(self, content):
        assert self.IsListLike(), "Only ListLike PyObj supports append()"
        self.__value__.append(content)
        return self
    def SetResolveBase(self, value=True):
        if value:
            self.__IsResolveBase__ = True
        else:
            if hasattr(self, "__IsResolveBase__"):
                delattr(self, "__IsResolveBase__")
    def SetResolveBaseRecur(self):
        self.SetResolveBase()
        if hasattr(self, "Modules"):
            #self.Modules.SetResolveBase()
            for ModuleName, ModuleParam in self.Modules.ListAttrsAndValues():
                if isinstance(ModuleParam, PyObj):
                    ModuleParam.SetResolveBaseRecur()
    def IsResolveBase(self):
        if hasattr(self, "__IsResolveBase__"):
            if self.__IsResolveBase__==True or self.__IsResolveBase__ in ["here"]:
                return True
        return False
    def Keys(self):
        return list(self.__dict__.keys())
    def Values(self):
        return self.__dict__.values()
PyObj.append = PyObj.Append
PyObj.items = PyObj.ListAttrsAndValues = PyObj.Items
PyObj.ListKeys = PyObj.keys = PyObj.Keys
PyObj.values = PyObj.Values
