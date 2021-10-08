import re
import json
import json5
from typing import List

import utils_torch
from utils_torch.python import CheckIsLegalPyName
from utils_torch.attrs import *
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
def PyObj2DataFile(Obj, FilePath):
    DataObj = PyObj2DataObj(Obj)
    # if "VerticesNp" in DataObj:
    #     print("aaa")
    JsonObj2DataFile(DataObj, FilePath)

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
    # if "VerticesNp" in DataObj:
    #     print("aaa")
    return JsonObj2PyObj(DataObj)