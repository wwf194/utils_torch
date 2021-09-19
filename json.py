import re
import json
import json5
from typing import List

import utils_torch
from utils_torch.python import CheckIsLegalPyName
from utils_torch.attrs import *
# from utils_torch.utils import ListAttrs # leads to recurrent reference.

def EmptyPyObj():
    return PyObj()

def CheckIsPyObj(Obj):
    if not IsPyObj(Obj):
        raise Exception("Obj is not PyObj, but %s"%type(Obj))

def IsPyObj(Obj):
    return isinstance(Obj, PyObj)

class PyObj(object):
    def __init__(self, param=None):
        if param is not None:
            if type(param) is dict:
                self.FromDict(param)
            elif type(param) is list:
                self.FromList(param)
            elif type(param) is str:
                param = json.loads(param)
            else:
                raise Exception()
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
            if key in ["Initialize.Method"]:
                print("aaa")
            if "." in key:
                keys = key.split(".")
            else:
                keys = [key]
            CheckIsLegalPyName(key[0])
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
    # def SetAttr(self, obj, attr, value):
    #     if type(value) is dict:
    #         if hasattr(obj, attr) and isinstance(getattr(obj, attr), PyObj):
    #             getattr(obj, attr).FromDict(value)
    #         else: # overwrite
    #             setattr(obj, attr, PyObj(value))
    #     elif type(value) is list:
    #         # always overwrite
    #         setattr(obj, attr, obj.FromList(value))
    #     else:
    #         # alwayes overwrite
    #         setattr(obj, attr, value)
    def ProcessValue(self, value):
        if isinstance(value, dict):
            return PyObj(value)
        elif isinstance(value, list):
            return PyObj().FromList(value)
        elif type(value) is PyObj:
            return value
        else:
            return value

    def to_dict(self):
        d = {}
        for key, value in self.__dict__.items():
            if type(value) is PyObj:
                value = value.to_dict()
            d[key] = value
        return d
    def __repr__(self):
        return str(self.to_dict())
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    def __getitem__(self, key):
        return self.__dict__[key]

def JsonObj2PyObj(JsonObj):
    if isinstance(JsonObj, list):
        return PyObj().FromList(JsonObj)
    elif isinstance(JsonObj, dict):
        return PyObj().FromDict(JsonObj)
    else:
        raise Exception()

Dict2PyObj = JsonObj2PyObj

json_obj_to_object = JsonObj2PyObj

def JsonObj2JsonStr(JsonObj):
    return json.dumps(JsonObj, indent=4, sort_keys=False)

def PyObj2JsonObj(Obj):
    if isinstance(Obj, list) or isinstance(Obj, tuple):
        JsonObj = []
        for Index, Item in enumerate(Obj):
            JsonObj.append(PyObj2JsonObj(Item))
        return JsonObj
    elif isinstance(Obj, PyObj):
        JsonObj = {}
        for attr, value in Obj.__dict__.items():
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

def JsonStr2JsonFile(JsonStr, path):
    with open(path, "w") as f:
        f.write(JsonStr)

new_json_file = JsonStr2JsonFile

