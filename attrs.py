from utils_torch.json import PyObj
import utils_torch

def CheckAttrs(Obj, attrs=[], *args, **kw):
    if kw.get("value") is None:
        raise Exception()
    Value = GetAttrs(Obj, attrs, *args)
    if Value != kw["value"]:
        raise Exception("CheckAttrs: %s != %s"%(Value, kw["value"]))

def SetAttrs(Obj, attrs=[], *args, **kw):
    if kw.get("value") is None:
        attrs = _ParseAttrs(attrs, *args)
        if len(attrs) > 0:
            kw["value"] = attrs[-1]
            attrs = attrs[:-1]
            args = []
        else:
            raise Exception("SetAttrs: named parameter value must be given.")

    kw["WriteDefault"] = True
    kw["default"] = kw["value"]
    EnsureAttrs(Obj, attrs, *args, **kw)

def RemoveAttrs(Obj, attrs, *args):
    attrs = _ParseAttrs(attrs, *args)
    if not HasAttrs(Obj, attrs, *args):
        return False
    else:
        count = 0
        for attr in attrs:
            if count < len(attrs) - 1:
                if isinstance(Obj, dict) or isinstance(Obj, list):
                    Obj = Obj[attr]
                elif isinstance(Obj, object):
                    Obj = getattr(Obj, attr)
                else:
                    raise Exception() # to be implemented
            else:
                if isinstance(Obj, dict):
                    Obj.pop(attr)
                elif isinstance(Obj, list):
                    del Obj[attr]
                elif isinstance(Obj, object):
                    delattr(Obj, attr)
                else:
                    raise Exception() # to be implemented
            count += 1

def MatchAttrs(Obj, attrs=None, *args, **kw):
    if kw.get("value") is None and len(args)==0:
        raise Exception("MatchAttrs: named parameter value must be given.")
    # return True if and only if Obj.attrs exists and equals to value
    if attrs is None:
        return Obj==kw["value"]
    else:
        if HasAttrs(Obj, attrs, *args):
            if GetAttrs(Obj, attrs, *args)==kw["value"]:
                return True
        return False

def EnsureAttrs(Obj, attrs=[], *args, **kw):
    attrs = _ParseAttrs(attrs, *args)
    if kw.get("default") is None:
        if kw.get("value") is not None:
            kw["default"] = kw["value"]
        else:
            if len(attrs)>0:
                kw["default"] = attrs[-1]
                attrs = attrs[:-1]
            else:
                raise Exception()
    default = kw["default"]
    count = 0

    if len(attrs) == 0:
        if isinstance(Obj, PyObj):
            if kw.get("WriteDefault")==True:
                setattr(Obj, "__value__", default)
        return

    parent, parentAttr = None, None
    for index, attr in enumerate(attrs):
        if index < len(attrs) - 1:
            if hasattr(Obj, "__dict__"):
                parent = Obj
                parentAttr = attr
                if hasattr(Obj, attr):
                    Obj = getattr(Obj, attr)
                else:
                    setattr(Obj, attr, utils_torch.PyObj())
                    Obj = getattr(Obj, attr)               
            else:
                SetAttr(parent, parentAttr, utils_torch.json.PyObj({"__value__": Obj}))
                Obj = getattr(parent, parentAttr)
                parent = Obj
                parentAttr = attr
                setattr(Obj, attr, utils_torch.PyObj())
                Obj = getattr(Obj, attr)                    
        else:
            if hasattr(Obj, "__dict__"):
                if hasattr(Obj, attr):
                    value = getattr(Obj, attr)
                    if value is not None: # Obj already has a not None attribute
                        if kw.get("WriteDefault")==True:
                            setattr(Obj, attr, default)
                    else:
                        setattr(Obj, attr, default)
                else:
                    setattr(Obj, attr, default)
            else:
                if kw.get("WriteDefault")==True:
                    if parent is None:
                        raise Exception("EnsureAttrs: Cannot redirect parent attribute.")
                    SetAttr(parent, parentAttr, 
                        utils_torch.PyObj({
                            "__value__": Obj,
                        }))
                    
                    Obj = getattr(parent, parentAttr)
                    setattr(Obj, attr, default)
        count += 1

ensure_attrs = EnsureAttrs

def SetAttr(Obj, Attr, Value):
    if isinstance(Obj, list):
        if isinstance(Attr, str):
            Attr = int(Attr)
        Obj[Attr] = Value
    elif isinstance(Obj, dict):
        Obj[Attr] = Value
    elif hasattr(Obj, "__dict__"):
        setattr(Obj, Attr, Value)
    else:
        raise Exception()

def HasAttrs(Obj, attrs, *args, false_if_none=True):
    attrs = _ParseAttrs(attrs, *args)
    for attr in attrs:
        if hasattr(Obj, attr):
            Obj = getattr(Obj, attr)
        else:
            return False
    if false_if_none:
        if Obj is None:
            return False
        else:
            return True
    else:
        return True

has_attrs = HasAttrs

def ListAttrs(Obj):
    return [(attr, value) for attr, value in Obj.__dict__.items()]

def ListValues(Obj):
    return Obj.__dict__.values()

list_attrs = ListAttrs

def GetAttrs(Obj, attrs=[], *args):
    attrs = _ParseAttrs(attrs, *args)
    attrs_reached = []
    for attr in attrs:
        attrs_reached.append(attr)
        if isinstance(Obj, dict):
            Obj = Obj[attr]
        elif isinstance(attr, int): # Obj is a list
            Obj = Obj[attr]
        elif isinstance(attr, object):
            if isinstance(attr, str): # Obj is an object
                if hasattr(Obj, attr):
                    Obj = getattr(Obj, attr)
                else:
                    raise Exception("GetAttrs: non-existent attr: %s"%(".".join(attrs_reached)))
        else:
            raise Exception("GetAttrs: invalid attr type: %s"%(".".join(attrs_reached)))
    
    if isinstance(Obj, dict) and Obj.get("__value__") is not None:
        return Obj["__value__"]
    if hasattr(Obj, "__dict__") and hasattr(Obj, "__value__"):
        return Obj.__value__
    return Obj

Getattrs = GetAttrs

def _ParseAttrs(attrs, *args):
    if isinstance(attrs, list):
        attrs_origin = [*attrs, *args]
    elif isinstance(attrs, str):
        attrs_origin = [attrs, *args]
    else:
        raise Exception()
    attrs = []
    for attr in attrs_origin:
        if isinstance(attr, str):
            attrs = [*attrs, *attr.split(".")]
        else:
            attrs = [*attrs, attr]
    return attrs

_parse_attrs = _ParseAttrs