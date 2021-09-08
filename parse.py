from utils_torch.attrs import *
from utils_torch.json import *

def ParseShape(Shape):
    if isinstance(Shape, int):
        Shape = (Shape)
    elif isinstance(Shape, list) or isinstance(Shape, tuple):
        return Shape
    else:
        raise Exception()

def Redirect2PyObj(PyObj, PyObjRef=None):
    # Traverse Attributes in PyObj, recursively
    # If an Attribute Value is str and begins with &, redirect this Attribute to Attribute in PyObjRef it points to.
    if PyObjRef is None:
        PyObjRef = PyObj
    _Redirect2PyObj(PyObj, PyObjRef, None, None)

def _Redirect2PyObj(PyObj, PyObjRef, parent, Attr):
    if isinstance(PyObj, dict):
        for key, Value in PyObj.items():
            _Redirect2PyObj(Value, PyObjRef, PyObj, key)
    elif isinstance(PyObj, list):
        for Index, Item in enumerate(list):
            _Redirect2PyObj(Item, PyObjRef, PyObj, Index)
    elif isinstance(PyObj, str):
        if PyObj[0]=="&":
            ValueRef = GetAttrs(PyObjRef, PyObj[1:])
            SetAttrs(parent, Attr, Value=ValueRef)
    elif hasattr(PyObj, "__dict__"):
        for Attr, Value in ListAttrs(PyObj):
            _Redirect2PyObj(getattr(PyObj, Attr), PyObjRef, PyObj, Attr)
    else:
        pass


''' Deprecated. No longer used by ParseJsonObj
def _ParseJsonObj(obj, root, Attrs, parent):
    if isinstance(obj, list):
        for index, Item in enumerate(obj):
            _ParseJsonObj(Item, root, Attrs + [index], obj)
    elif isinstance(obj, dict):
        for key, Value in obj.items():
            _ParseJsonObj(Value, root, Attrs + [key], obj)
    elif isinstance(obj, str):
        if obj[0]=="$":
            parent[Attrs[-1]] = GetAttrs(root, obj[1:])
    else:
        #raise Exception("_ParseJsonObj: Invalid type: %s"%type(obj))
        pass
    return obj
'''

def ParseFunctionArgs(FunctionArgs, ContextInfo):
    if type(FunctionArgs) is not list:
        raise Exception()
    PositionalArgs = []
    KeyWordArgs = {}
    for Arg in FunctionArgs:
        if isinstance(Arg, str) and "=" in Arg:
            Arg = Arg.split("=")
            if not len(Arg) == 2:
                raise Exception()
            Key = utils_torch.RemoveHeadTailWhiteChars(Arg[0])
            Value = utils_torch.RemoveHeadTailWhiteChars(Arg[1])
            KeyWordArgs[Key] = ParseFunctionArg(Value, ContextInfo)
        else:
            ArgParsed = ParseFunctionArg(Arg, ContextInfo)
            PositionalArgs.append(ArgParsed)
    return PositionalArgs, KeyWordArgs

def ParseFunctionArg(Arg, ContextInfo):
    if isinstance(Arg, str):
        if Arg.startswith("__") and Arg.endswith("__"):
            return GetAttrs(ContextInfo, Arg)
        elif "&" in Arg:
            #ArgParsed = ParseAttr(Arg, **utils_torch.json.PyObj2JsonObj(ContextInfo))
            return ParseAttrFromContextInfo(Arg, ContextInfo)
        else:
            return Arg
    else:
        return Arg

def ParseAttr(Str, **kw):
    ObjRoot = kw.setdefault("ObjRoot", None)
    ObjCurrent = kw.setdefault("ObjCurrent", None)
    if "&" in Str:
        sentence = Str.replace("&^", "ObjRoot.").replace("&", "ObjCurrent.")
        return eval(sentence)
    else:
        return Str

def ParseAttrFromContextInfo(Str, ContextInfo):
    EnsureAttrs(ContextInfo, "ObjRoot", None)
    EnsureAttrs(ContextInfo, "ObjCurrent", None)
    if "&" in Str:
        sentence = Str.replace("&^", "GetAttrs(ContextInfo, 'ObjRoot').").\
            replace("&", "GetAttrs(ContextInfo, 'ObjCurrent').")
        return eval(sentence)
    else:
        return Str

def ParsePyObjDynamic(Obj, **kw):
    ObjParsed = PyObj()
    if kw.get("ObjRefList") is not None:
        return ParsePyObjDynamicWithMultipleRefs(Obj, **kw)
    else:
        return _ParsePyObjDynamic(Obj, ObjParsed, [], **kw)

def _ParsePyObjDynamic(Obj, ObjParsed, Attrs, **kw):
    if isinstance(Obj, list):
        ObjParsed = []
        for Index, Item in enumerate(Obj):
            ObjParsed.append(_ParsePyObjDynamic(Item, [*Attrs, Index], Obj, **kw))
    elif isinstance(Obj, dict):
        ObjParsed = {}
        for Key, Value in Obj.items():
            ObjParsed[Key] = _ParsePyObjDynamic(Value, ObjParsed, [*Attrs, Key], **kw)
    elif hasattr(Obj, "__dict__"):
        ObjParsed = PyObj()
        for Attr, Value in Obj.__dict__.items():
            setattr(ObjParsed, Attr, _ParsePyObjDynamic(Value, Obj, [*Attrs, Attr], **kw))
    elif isinstance(Obj, str) and "&" in Obj:
        sentence = Obj
        if "&^" in sentence:
            sentence = sentence.replace("&^", "kw['ObjRoot'].")
        if "&" in sentence:
            sentence = sentence.replace("&", "kw['ObjCurrent'].")
        
        # Some Tricks
        if "|-->" in sentence:
            ObjParsed = Obj
        else:
            try:
                ObjParsed = eval(sentence)
            except Exception:
                ObjParsed = Obj
    else:
        ObjParsed = Obj
    return ObjParsed

def ParsePyObjDynamicWithMultipleRefs(PyObj, ObjRefList, **kw):
    # Traverse Attributes in PyObj, recursively
    # If an Attribute Value is str and begins with &, redirect this Attribute to Attribute in PyObjRef it points to.
    if not isinstance(ObjRefList, list):
        if hasattr(ObjRefList, "__dict__"):
            ObjRefList = [ObjRefList]
        else:
            raise Exception()
    return _ParsePyObjDynamicWithMultipleRefs(PyObj, ObjRefList, None, **kw)

def _ParsePyObjDynamicWithMultipleRefs(Obj, ObjRefList, Attr, **kw):
    if isinstance(Obj, dict):
        ObjParsed = {}
        for Key, Value in Obj.items():
            ObjParsed[Key] = _ParsePyObjDynamicWithMultipleRefs(Value, ObjRefList, Key, **kw)
    elif isinstance(Obj, list):
        ObjParsed = []
        for Index, Item in enumerate(Obj):
            ObjParsed.append(_ParsePyObjDynamicWithMultipleRefs(Item, ObjRefList, Index, **kw))
    elif hasattr(Obj, "__dict__"):
        ObjParsed = PyObj()
        for Attr, Value in ListAttrs(Obj):
            setattr(ObjParsed, Attr, _ParsePyObjDynamicWithMultipleRefs(getattr(Obj, Attr), ObjRefList, Attr, **kw))
    elif isinstance(Obj, str):
        sentence = Obj
        if "&" in sentence:
            success = False
            for ObjRef in ObjRefList:
                try:
                    if "&^" in sentence:
                        sentence = sentence.replace("&^", "kw['ObjRoot'].")
                    ObjParsed = eval(sentence.replace("&", "kw['ObjRef']."))
                    success = True
                    break
                except Exception:
                    utils_torch.AddLog("Failed to redirect to current PyObjRef. Try redirecting to next PyObjRef.")
            if not success:
                utils_torch.AddWarning("Failed to redirect to any PyObjRef in given ObjRefList.")
                ObjParsed = Obj
        else:
            ObjParsed = sentence
    else:
        ObjParsed = Obj
    return ObjParsed

def ProcessPyObj(Obj, Function=lambda x:(x, True), **kw):
    ObjParsed = PyObj()
    return _ProcessPyObj(Obj, Function, ObjParsed, [], **kw)

def _ProcessPyObj(Obj, Function, ObjParsed, Attrs, **kw):
    Obj, ContinueParse = Function(Obj)
    if not ContinueParse:
        return Obj
    if isinstance(Obj, list):
        ObjParsed = []
        for Index, Item in enumerate(Obj):
            ObjParsed.append(_ParsePyObj(Item, [*Attrs, Index], Obj), **kw)
    elif isinstance(Obj, dict):
        ObjParsed = {}
        for Key, Value in Obj.items():
            ObjParsed[Key] = _ParsePyObj(Value, ObjParsed, [*Attrs, Key], **kw)
    elif hasattr(Obj, "__dict__"):
        ObjParsed = PyObj()
        for Attr, Value in Obj.__dict__.items():
            setattr(ObjParsed, Attr, _ParsePyObj(Value, Obj, [*Attrs, Attr]))
    else:
        ObjParsed = Obj
    return ObjParsed

def ParsePyObjStatic(PyObj, ObjRoot=None, ObjCurrent=None):
    _ParsePyObj(PyObj, ObjRoot=ObjRoot, Attrs=[], parent=None, ObjCurrent=ObjCurrent)
    return PyObj
ParsePyObj = ParsePyObjStatic

def _ParsePyObjStatic(obj, ObjRoot, Attrs, parent, ObjCurrent="root."):
    if hasattr(obj, "__DollarPath__"):
        ObjCurrent = getattr(obj, "__DollarPath__")
    if isinstance(obj, list):
        for index, Item in enumerate(obj):
            _ParsePyObj(Item, ObjRoot, Attrs + [index], obj, ObjCurrent)
    elif isinstance(obj, dict):
        for key, Value in obj.items():
            _ParsePyObj(Value, ObjRoot, Attrs + [key], obj, ObjCurrent)
    elif isinstance(obj, str):
        sentence = obj
        while type(sentence) is str and ("$" in sentence or "^" in sentence) and ("&" not in sentence):
            sentence = sentence.replace("$", "ObjCurrent.").replace("^", "ObjRoot.")
            try:
                sentence = eval(sentence)
            except Exception:
               utils_torch.AddLog("Exception when running %s"%sentence)
               raise Exception()
        if isinstance(sentence, str) and sentence.startswith("#"):
            sentence = eval(sentence[1:])
        parent[Attrs[-1]] = sentence
    elif hasattr(obj, "__dict__"):
        for Attr, Value in obj.__dict__.items():
            _ParsePyObj(Value, ObjRoot, [*Attrs, Attr], obj, ObjCurrent)
    else:
        #raise Exception("_ParseJsonObj: Invalid type: %s"%type(obj))
        pass
    return obj

_ParsePyObj = _ParsePyObjStatic

def ParseParamJsonObj(JsonObj, overwrite=True):
    PyObj = JsonObj2PyObj(JsonObj)
    for name, obj in utils_torch.ListAttrs(PyObj):
        setattr(obj, "__DollarPath__", "root.%s."%name)
    json_dicts_Parsed = ParsePyObj(PyObj)
    for Value in json_dicts_Parsed.Values():
        Value.pop("__DollarPath__")
    for name, obj in utils_torch.ListAttrs(PyObj):
        delattr(obj, "__DollarPath__")
    return json_dicts_Parsed

def ParseParamPyObjStatic(PyObj, overwrite=True):
    for name, obj in utils_torch.ListAttrs(PyObj):
        setattr(obj, "__DollarPath__", "root.%s."%name)
    PyObjParsed = ParsePyObjStatic(PyObj)
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
    for Value in json_dicts_parsed.Values():
        Value.pop("__DollarPath__")
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

def ParseRoutingsStr(Routings):
    RoutingsParsed = []
    if not isinstance(Routings, list):
        _ParseRoutingStr(Routings)
    for Index, Routing in enumerate(Routings):
        RoutingsParsed.append(_ParseRoutingStr(Routing))
    return RoutingsParsed

def _ParseRoutingStr(RoutingStr):
    Routing = re.sub(" ", "", RoutingStr) # remove all spaces
    param = utils_torch.json.EmptyPyObj()
    Routing = Routing.split("||")
    MainRouting = Routing[0] 
    if len(Routing) > 1:
        Params = Routing[1:]
    else:
        Params = []

    MainRouting = MainRouting.split("|-->")
    if len(MainRouting) != 3:
        raise Exception("Routing Must Be In Form Of ... |--> ... |--> ...")
    In = MainRouting[0]
    Module = MainRouting[1]
    Out = MainRouting[2]

    param.In = In.rstrip(",").split(",")
    param.Out = Out.rstrip(",").split(",")
    param.Module = Module

    for Param in Params:
        Param = Param.split("=")
        if len(Param)!=2:
            raise Exception()
        attr, value = Param[0], Param[1]
        if attr in ["repeat"]:
            attr = "RepeatTime"
        setattr(param, attr, value)
    EnsureAttrs(param, "RepeatTime", value=1)

    param.DynamicParseAttrs = []
    for attr, value in ListAttrs(param):
        if isinstance(value, str):
            #if "%" in value:
            if value[0]=="%": # Dynamic Parse
                param.DynamicParseAttrs.append(attr)
    return param

def FilterFromPyObj(PyObj, Keys):
    List = []
    for Key in Keys:
        Value = GetAttrs(PyObj, Key)
        List.append(Value)
    return List

def Register2PyObj(Obj, PyObj, Names):
    if isinstance(Obj, list):
        RegisterList2PyObj(Obj, PyObj, Names)
    elif isinstance(Obj, dict):
        RegisterDict2PyObj(Obj, PyObj, Names)
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

def ParseRouters(Routers):
    RouterParsed = []
    for Router in Routers:
        RouterParsed.append(ParseRouter(Router))
    return RouterParsed

def ParseRouter(Router, ObjRefList=[], **kw):
    SetAttrs(Router, "Routings", value=ParseRoutingsStr(Router.Routings))
    return utils_torch.parse.ParsePyObjDynamic(Router, ObjRefList=ObjRefList, **kw)

def ParseRoutingDynamic(Routing, States):
    #utils_torch.parse.RedirectPyObj(Routing, States)
    for attrs in Routing.DynamicParseAttrs:
        value = GetAttrs(Routing, attrs)    
        value = re.sub("([%]\w+)", lambda matchedStr:"getattr(States, %s)"%matchedStr[1:], value)
        SetAttrs(Routing, attrs, eval(value))
    return Routing

def SeparateArgs(ArgsString):
    # ArgsString = ArgsString.strip() # remove empty chars at front and end.
    # ArgsString.rstrip(",")
    # Args = ArgsString.split(",")
    # for Arg in Args:
    #     Arg = Arg.strip()
    ArgsString = re.sub(" ", "", ArgsString) # remove all empty spaces.
    ArgsString = ArgsString.rstrip(",")
    Args = ArgsString.split(",")

    if len(Args)==1 and Args[0]=="":
        return []
    else:
        return Args