from utils_torch.attrs import *
from utils_torch.json import *

def Redirect2PyObj(PyObj, PyObjRef=None):
    # Traverse attributes in PyObj, recursively
    # If an attribute value is str and begins with &, redirect this attribute to attribute in PyObjRef it points to.
    if PyObjRef is None:
        PyObjRef = PyObj
    _Redirect2PyObj(PyObj, PyObjRef, None, None)

def _Redirect2PyObj(PyObj, PyObjRef, parent, attr):
    if isinstance(PyObj, dict):
        for key, value in PyObj.items():
            _Redirect2PyObj(value, PyObjRef, PyObj, key)
    elif isinstance(PyObj, list):
        for Index, Item in enumerate(list):
            _Redirect2PyObj(Item, PyObjRef, PyObj, Index)
    elif isinstance(PyObj, str):
        if PyObj[0]=="&":
            valueRef = GetAttrs(PyObjRef, PyObj[1:])
            SetAttrs(parent, attr, value=valueRef)
    elif hasattr(PyObj, "__dict__"):
        for attr, value in ListAttrs(PyObj):
            _Redirect2PyObj(getattr(PyObj, attr), PyObjRef, PyObj, attr)
    else:
        pass

def Redirect2PyObjList(PyObj, PyObjRefs):
    # Traverse attributes in PyObj, recursively
    # If an attribute value is str and begins with &, redirect this attribute to attribute in PyObjRef it points to.
    if not isinstance(PyObjRefs, list):
        raise Exception()
    _Redirect2PyObjList(PyObj, PyObjRefs, None, None)

def _Redirect2PyObjList(PyObj, PyObjRefs, parent, attr):
    if isinstance(PyObj, dict):
        for key, value in PyObj.items():
            _Redirect2PyObjList(value, PyObjRefs, PyObj, key)
    elif isinstance(PyObj, list):
        for Index, Item in enumerate(list):
            _Redirect2PyObjList(Item, PyObjRefs, PyObj, Index)
    elif isinstance(PyObj, str):
        if PyObj[0]=="&":
            success = False
            for PyObjRef in PyObjRefs:
                try:
                    valueRef = GetAttrs(PyObjRef, PyObj[1:])
                    success = True
                    break
                except Exception:
                    utils_torch.add_log("Failed to redirect to current PyObjRef. Try redirecting to next PyObjRef.")
            if not success:
                raise Exception("Failed to redirect to any PyObjRef in given PyObjRefs.")
            SetAttrs(parent, attr, value=valueRef)
    elif hasattr(PyObj, "__dict__"):
        for attr, value in ListAttrs(PyObj):
            _Redirect2PyObjList(getattr(PyObj, attr), PyObjRefs, PyObj, attr)
    else:
        pass

''' Deprecated. No longer used by ParseJsonObj
def _ParseJsonObj(obj, root, attrs, parent):
    if isinstance(obj, list):
        for index, item in enumerate(obj):
            _ParseJsonObj(item, root, attrs + [index], obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _ParseJsonObj(value, root, attrs + [key], obj)
    elif isinstance(obj, str):
        if obj[0]=="$":
            parent[attrs[-1]] = GetAttrs(root, obj[1:])
    else:
        #raise Exception("_ParseJsonObj: Invalid type: %s"%type(obj))
        pass
    return obj
'''

def ParsePyObj(PyObj):
    _ParsePyObj(PyObj, root=PyObj, attrs=[], parent=None)
    return PyObj

def _ParsePyObj(obj, root, attrs, parent, base_path="root."):
    if hasattr(obj, "__DollarPath__"):
        base_path = getattr(obj, "__DollarPath__")
    if isinstance(obj, list):
        for index, item in enumerate(obj):
            _ParsePyObj(item, root, attrs + [index], obj, base_path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _ParsePyObj(value, root, attrs + [key], obj, base_path)
    elif isinstance(obj, str):
        sentence = obj
        while type(sentence) is str and ("$" in sentence or "^" in sentence):
            sentence = sentence.replace("$", base_path).replace("^", "root.")
            #try:
            sentence = eval(sentence)
            #except Exception:
            #    utils_torch.add_log("Exception when running %s"%sentence)
            #    raise Exception()
        if isinstance(sentence, str) and sentence.startswith("#"):
            sentence = eval(sentence[1:])
        parent[attrs[-1]] = sentence
    elif hasattr(obj, "__dict__"):
        for attr, value in obj.__dict__.items():
            _ParsePyObj(value, root, [*attrs, attr], obj, base_path)
    else:
        #raise Exception("_ParseJsonObj: Invalid type: %s"%type(obj))
        pass
    return obj

_ParsePyObj = _ParsePyObj

def ParseParamJsonObj(JsonObj, overwrite=True):
    PyObj = JsonObj2PyObj(JsonObj)
    for name, obj in utils_torch.ListAttrs(PyObj):
        setattr(obj, "__DollarPath__", "root.%s."%name)
    json_dicts_Parsed = ParsePyObj(PyObj)
    for value in json_dicts_Parsed.values():
        value.pop("__DollarPath__")
    for name, obj in utils_torch.ListAttrs(PyObj):
        delattr(obj, "__DollarPath__")
    return json_dicts_Parsed

def ParseParamPyObj(PyObj, overwrite=True):
    for name, obj in utils_torch.ListAttrs(PyObj):
        setattr(obj, "__DollarPath__", "root.%s."%name)
    PyObjParsed = ParsePyObj(PyObj)
    for name, obj in utils_torch.ListAttrs(PyObjParsed):
        delattr(obj, "__DollarPath__")
    return PyObjParsed

def ParseJsonObj(JsonObj): # obj can either be dict or list.
    PyObj = JsonObj2PyObj(JsonObj)
    ParsePyObj(PyObj)
    return PyObj

JsonObj2ParsedJsonObj = ParseJsonObj


def parse_param_json_obj(json_obj, overwrite=True):
    py_obj = JsonObj2PyObj(json_obj)
    for name, obj in utils_torch.ListAttrs(py_obj):
        setattr(obj, "__DollarPath__", "root.%s."%name)
    json_dicts_parsed = ParsePyObj(py_obj)
    for value in json_dicts_parsed.values():
        value.pop("__DollarPath__")
    for name, obj in utils_torch.ListAttrs(py_obj):
        delattr(obj, "__DollarPath__")
    return json_dicts_parsed

def parse_param_py_obj(py_obj, overwrite=True):
    for name, obj in utils_torch.ListAttrs(py_obj):
        setattr(obj, "__DollarPath__", "root.%s."%name)
    py_obj_parsed = ParsePyObj(py_obj)
    for name, obj in utils_torch.ListAttrs(py_obj_parsed):
        delattr(obj, "__DollarPath__")
    return py_obj_parsed

def parse_json_obj(json_obj): # obj can either be dict or list.
    py_obj = JsonObj2PyObj(json_obj)
    ParsePyObj(py_obj)
    return py_obj

JsonObj2ParsedJsonObj = parse_json_obj

def JsonObj2ParsedPyObj(json_obj):
    return JsonObj2PyObj(JsonObj2ParsedJsonObj(json_obj))

