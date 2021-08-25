
import utils_torch
# from utils_torch.utils import ListAttrs # leads to recurrent reference.
import json
import json5

class PyObj(object):
    def __init__(self, param=None):
        if param is not None:
            if type(param) is dict:
                self.from_dict(param)
            elif type(param) is list:
                self.from_list(param)
            elif type(param) is str:
                param = json.loads(param)
            else:
                raise Exception()
    def from_list(self, list):
        for index, item in enumerate(list):
            if type(item) is dict:
                list[index] = PyObj(item)
            elif type(item) is list:
                self.from_list(item)
            else:
                pass
        return list
    def from_dict(self, dict_):
        self.__dict__ = {}
        for key, value in dict_.items():
            if "." in key:
                keys = key.split(".")
                checkIsLegalPyName(key[0])
                for index in range(len(keys)):
                    key = keys[index]
                    obj = self
                    if hasattr(obj, key):
                        obj = getattr(self, key)
                    else:
                        if index == len(keys) - 1:
                            setattr(obj, key, PyObj({
                                keys[index]: value
                            }))
                        else:
                            setattr(obj, key, PyObj({
                                ".".join(keys[index + 1:]): value
                            }))
            else:
                checkIsLegalPyName(key)
                if type(value) is dict:
                    value = PyObj(value)
                elif type(value) is list:
                    self.from_list(value)
                else:
                    pass
                setattr(self, key, value)
        return self
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

def JsonObj2PyObj(json_obj):
    if isinstance(json_obj, list):
        return PyObj().from_list(json_obj)
    elif isinstance(json_obj, dict):
        return PyObj().from_dict(json_obj)
    else:
        raise Exception()
json_obj_to_object = JsonObj2PyObj
JsonObj2PyObj = JsonObj2PyObj

def JsonObj2JsonStr(json_obj):
    return json5.dumps(json_obj)

def PyObj2JsonObj(obj):
    return json.loads(PyObj2JsonStr(obj))

def PyObj2JsonFile(obj, path):
    # json_obj = PyObj2JsonObj(obj)
    # JsonStr = JsonObj2JsonStr(json_obj)
    JsonStr = PyObj2JsonStr(obj)
    JsonStr2JsonFile(JsonStr, path)

def JsonStr2JsonFile(JsonStr, path):    
    if path.endswith(".jsonc") or path.endswith(".json"):
        pass
    else:
        path += ".jsnonc"
    with open(path, "w") as f:
        f.write(JsonStr)

object_to_json_obj = PyObj2JsonObj

def PyObj2JsonStr(obj):
    # return json.dumps(obj.__dict__, cls=change_type,indent=4)
    # why default=lambda o: o.__dict__?
    return json.dumps(obj, default=lambda o: o.__dict__, sort_keys=False, indent=4)

def JsonStr2JsonObj(JsonStr):
    return json.loads(JsonStr)

def JsonStr2PyObj(JsonStr):
    JsonObj = JsonStr2JsonObj(JsonStr)
    return JsonObj2PyObj(JsonObj)
    # return json.loads(JsonStr, object_hook=lambda d: SimpleNamespace(**d))
JsonStr_to_object = JsonStr2PyObj

def JsonFile2PyObj(FilePath):
    return JsonObj2PyObj(JsonFile2JsonObj(FilePath))

def JsonFile2JsonObj(FilePath):
    with open(FilePath, "r") as f:
        json_dict = json5.load(f)
    return json_dict
load_json_file = JsonFile2JsonObj

def JsonObj2JsonFile(JsonObj, FilePath):
    return JsonStr2JsonFile(JsonObj2JsonStr(JsonObj), FilePath)

def JsonStr2JsonFile(JsonStr, path):
    with open(path, "w") as f:
        f.write(JsonStr)

new_json_file = JsonStr2JsonFile


def isLegalPyName(name):
    if name=="":
        return False
    if name[0].isalpha() or name[0] == '_':
        for i in name[1:]:
            if not (i.isalnum() or i == '_'):
                return False
        else:
            return True
    else:
        return False

def checkIsLegalPyName(name):
    if not isLegalPyName(name):
        raise Exception("%s is not a legal python name."%name)