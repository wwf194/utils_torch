
import utils_torch

def CheckAttrs(obj, attrs, *args, **kw):
    if kw.get("value") is None:
        raise Exception()
    if GetAttrs(obj, attrs, *args) != kw["value"]:
        raise Exception() 

def SetAttrs(obj, attrs, *args, **kw):
    if kw.get("value") is None:
        attrs = _ParseAttrs(attrs, *args)
        if len(attrs) > 1:
            kw["value"] = attrs[-1]
            attrs = attrs[:-1]
            args = []
        else:
            raise Exception("SetAttrs: named parameter value must be given.")
    else:
        default = kw["value"]
    kw["WriteDefault"] = True
    kw["default"] = kw["value"]
    EnsureAttrs(obj, attrs, *args, **kw)

def RemoveAttrs(obj, attrs, *args):
    attrs = _ParseAttrs(attrs, *args)
    if not HasAttrs(obj, attrs, *args):
        return False
    else:
        count = 0
        for attr in attrs:
            if count < len(attrs) - 1:
                if isinstance(obj, dict) or isinstance(obj, list):
                    obj = obj[attr]
                elif isinstance(obj, object):
                    obj = getattr(obj, attr)
                else:
                    raise Exception() # to be implemented
            else:
                if isinstance(obj, dict):
                    obj.pop(attr)
                elif isinstance(obj, list):
                    del obj[attr]
                elif isinstance(obj, object):
                    delattr(obj, attr)
                else:
                    raise Exception() # to be implemented
            count += 1

def MatchAttrs(obj, attrs=None, *args, **kw):
    if kw.get("value") is None and len(args)==0:
        raise Exception("MatchAttrs: named parameter value must be given.")
    # return True if and only if obj.attrs exists and equals to value
    if attrs is None:
        return obj==kw["value"]
    else:
        if HasAttrs(obj, attrs, *args):
            if GetAttrs(obj, attrs, *args)==kw["value"]:
                return True
        return False

def EnsureAttrs(obj, attrs, *args, **kw):
    if kw.get("default") is None:
        if kw.get("value") is not None:
            default = kw["value"]
        else:
            default = None
    else:
        default = kw["default"]

    attrs = _ParseAttrs(attrs, *args)
    #print(attrs)
    obj_root = obj
    count = 0
    for attr in attrs:
        if count < len(attrs) - 1:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                obj_empty = utils_torch.new_empty_object()
                setattr(obj, attr, obj_empty)
                obj = obj_empty
        else:
            if hasattr(obj, attr) and getattr(obj, attr) is not None:
                if kw.get("WriteDefault")==True:
                    setattr(obj, attr, default)
                else:
                    pass
            else:
                setattr(obj, attr, default)
        count += 1

ensure_attrs = EnsureAttrs

def HasAttrs(obj, attrs, *args, false_if_none=True):
    attrs = _ParseAttrs(attrs, *args)
    for attr in attrs:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            return False
    if false_if_none:
        if obj is None:
            return False
        else:
            return True
    else:
        return True

has_attrs = HasAttrs

def ListAttrs(obj):
    return [(attr, value) for attr, value in obj.__dict__.items()]

list_attrs = ListAttrs

def GetAttrs(obj, attrs, *args):
    attrs = _ParseAttrs(attrs, *args)
    attrs_reached = []
    for attr in attrs:
        attrs_reached.append(attr)
        if isinstance(obj, dict):
            obj = obj[attr]
        elif isinstance(attr, int): # obj is a list
            obj = obj[attr]
        elif isinstance(attr, object):
            if isinstance(attr, str): # obj is an object
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    raise Exception("GetAttrs: non-existent attr: %s"%(".".join(attrs_reached)))
        else:
            raise Exception("GetAttrs: invalid attr type: %s"%(".".join(attrs_reached)))
    return obj

get_attrs = GetAttrs

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