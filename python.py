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

new_empty_object = EmptyPyObj