import re
import json
import json5
from typing import List

import utils_torch
from utils_torch.python import CheckIsLegalPyName
from utils_torch.attrs import *
# from utils_torch.utils import ListAttrs # leads to recurrent reference.


def JsonObj2PyObj(JsonObj):
    if isinstance(JsonObj, list):
        return utils_torch.EmptyPyObj().FromList(JsonObj)
    elif isinstance(JsonObj, dict):
        return utils_torch.EmptyPyObj().FromDict(JsonObj)
    else:
        raise Exception()

Dict2PyObj = JsonObj2PyObj

json_obj_to_object = JsonObj2PyObj

def JsonObj2JsonStr(JsonObj):
    return json.dumps(JsonObj, indent=4, sort_keys=False)

def PyObj2JsonObj(Obj):
    if isinstance(Obj, utils_torch.PyObj) and Obj.IsListLike():
        Obj = Obj.ToList()
    if isinstance(Obj, list) or isinstance(Obj, tuple):
        JsonObj = []
        for Item in Obj:
            JsonObj.append(PyObj2JsonObj(Item))
        return JsonObj
    elif isinstance(Obj, utils_torch.PyObj):
        JsonObj = {}
        for attr, value in ListAttrsAndValues(Obj, Exceptions=[]):
            if attr in ["__ResolveRef__"]:
                continue
            JsonObj[attr] = PyObj2JsonObj(value)
        return JsonObj
    elif isinstance(Obj, dict):
        JsonObj = {}
        for key, value in Obj.items():
            JsonObj[key] = PyObj2JsonObj(value)
        return JsonObj   
    elif type(Obj) in [str, int, bool, float]:
        return Obj
    else:
        return "UnserializableObject"
    #return json.loads(PyObj2JsonStr(obj))

def PyObj2JsonFile(obj, path):
    # JsonObj = PyObj2JsonObj(obj)
    # JsonStr = JsonObj2JsonStr(JsonObj)
    JsonStr = PyObj2JsonStr(obj)
    JsonStr2JsonFile(JsonStr, path)

def JsonStr2JsonFile(JsonStr, FilePath):    
    if FilePath.endswith(".jsonc") or FilePath.endswith(".json"):
        pass
    else:
        FilePath += ".jsnonc"
    utils_torch.EnsureFileDir(FilePath)
    with open(FilePath, "w") as f:
        f.write(JsonStr)

object_to_json_obj = PyObj2JsonObj

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
    #return json.loads(JsonStr)
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
    return JsonObj2PyObj(JsonFile2JsonObj(FilePath))

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

load_json_file = JsonFile2JsonObj

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