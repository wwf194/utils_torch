def FilterDictFromPyObj(keys, PyObj):
    Dict = {}
    for key in keys:
        Dict[key] = getattr(PyObj, key)
    return Dict

def RegisterObj2PyObj(Obj, PyObj, names=None):
    if isinstance(Obj, list):
        RegisterList2PyObj(Obj, PyObj, names)
    elif isinstance(Obj, dict):
        RegisterDict2PyObj(Obj, PyObj, names)
    else:
        raise Exception()

def RegisterDict2PyObj(Dict, PyObj, keys=None):
    if keys is None:
        raise Exception()
    for key in keys:
        setattr(PyObj, key, Dict[key])

def RegisterList2PyObj(List, PyObj, attrs=None):
    if len(List)!=len(attrs):
        raise Exception()
    for index in range(len(List)):
        setattr(PyObj, attrs[index], List[index])

def EmptyPyObj():
    return type('test', (), {})()

new_empty_object = EmptyPyObj