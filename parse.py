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

def ParseFunctionArgs(Args, ContextInfo):
    if type(Args) is not list:
        raise Exception()
    PositionalArgs = []
    KeyWordArgs = {}
    for Arg in Args:
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
    if not isinstance(Str, str):
        return Str
    EnsureAttrs(ContextInfo, "ObjRoot", None)
    EnsureAttrs(ContextInfo, "ObjCurrent", None)
    if "&" in Str:
        sentence = Str.replace("&^", "GetAttrs(ContextInfo, 'ObjRoot').").\
            replace("&", "GetAttrs(ContextInfo, 'ObjCurrent').")
        return eval(sentence)
    else:
        return Str

def ParsePyObjDynamic(Obj, RaiseFailedParse=False, **kw):
    utils_torch.json.CheckIsPyObj(Obj)
    ObjParsed = PyObj()
    if kw.get("ObjRefList") is not None:
        return ParsePyObjDynamicWithMultipleRefs(Obj, RaiseFailedParse=RaiseFailedParse, **kw)
    else:
        return _ParsePyObjDynamic(Obj, ObjParsed, [], RaiseFailedParse, **kw)

def _ParsePyObjDynamic(Obj, ObjParsed, Attrs, RaiseFailedParse, **kw):
    if isinstance(Obj, list):
        ObjParsed = []
        for Index, Item in enumerate(Obj):
            ObjParsed.append(_ParsePyObjDynamic(Item, [*Attrs, Index], Obj, RaiseFailedParse, **kw))
    elif isinstance(Obj, dict):
        ObjParsed = {}
        for Key, Value in Obj.items():
            ObjParsed[Key] = _ParsePyObjDynamic(Value, ObjParsed, [*Attrs, Key], RaiseFailedParse, **kw)
    elif isinstance(Obj, utils_torch.json.PyObj):
        ObjParsed = PyObj()
        for Attr, Value in Obj.__dict__.items():
            setattr(ObjParsed, Attr, _ParsePyObjDynamic(Value, Obj, [*Attrs, Attr], RaiseFailedParse, **kw))
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
            if RaiseFailedParse:
                ObjParsed = eval(sentence)
            else: 
                try:
                    ObjParsed = eval(sentence)
                except Exception:
                    utils_torch.AddWarning("Failed to run: %s"%sentence)
                    ObjParsed = Obj
    else:
        ObjParsed = Obj
    return ObjParsed

def ParsePyObjDynamicWithMultipleRefs(PyObj, RaiseFailedParse=False, **kw):
    ObjRefList = kw["ObjRefList"]
    # Traverse Attributes in PyObj, recursively
    # If an Attribute Value is str and begins with &, redirect this Attribute to Attribute in PyObjRef it points to.
    if not isinstance(ObjRefList, list):
        if hasattr(ObjRefList, "__dict__"):
            ObjRefList = [ObjRefList]
        else:
            raise Exception()
    kw["ObjRefList"] = ObjRefList
    return _ParsePyObjDynamicWithMultipleRefs(PyObj, None, RaiseFailedParse, **kw)

def _ParsePyObjDynamicWithMultipleRefs(Obj, Attr, RaiseFailedParse, **kw):
    ObjRoot = kw.get("ObjRoot")
    ObjRef = kw.get("ObjRef")
    ObjRefList = kw["ObjRefList"]
    if isinstance(Obj, dict):
        ObjParsed = {}
        for Key, Value in Obj.items():
            ObjParsed[Key] = _ParsePyObjDynamicWithMultipleRefs(Value, Key, RaiseFailedParse, **kw)
    elif isinstance(Obj, list):
        ObjParsed = []
        for Index, Item in enumerate(Obj):
            ObjParsed.append(_ParsePyObjDynamicWithMultipleRefs(Item, Index, RaiseFailedParse, **kw))
    elif isinstance(Obj, utils_torch.json.PyObj):
        ObjParsed = PyObj()
        for Attr, Value in ListAttrsAndValues(Obj):
            setattr(ObjParsed, Attr, _ParsePyObjDynamicWithMultipleRefs(getattr(Obj, Attr), Attr, RaiseFailedParse, **kw))
    elif isinstance(Obj, str):
        sentence = Obj
        if "&" in sentence:
            success = False
            if "&^" in sentence:
                sentence = sentence.replace("&^", "ObjRoot.")
                sentence = sentence.replace("&", "ObjRef.")
            for ObjRef in ObjRefList:
                try:
                    ObjParsed = eval(sentence)
                    success = True
                    break
                except Exception:
                    pass
                    #utils_torch.AddLog("Failed to resoolve to current PyObjRef. Try redirecting to next PyObjRef.")
            if not success:
                report = "Failed to resolve to any PyObjRef in given ObjRefList by running: %s"%sentence
                if RaiseFailedParse:
                    raise Exception(report)
                else:
                    utils_torch.AddWarning(report)
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
        ObjParsed = EmptyPyObj()
        for Attr, Value in Obj.__dict__.items():
            setattr(ObjParsed, Attr, _ParsePyObj(Value, Obj, [*Attrs, Attr]))
    else:
        ObjParsed = Obj
    return ObjParsed

def ParsePyObjStatic(PyObj, ObjRoot=None, ObjCurrent=None):
    CheckIsPyObj(PyObj)
    return _ParsePyObjStatic(PyObj, Attrs=[], ObjRoot=ObjRoot, ObjCurrent=ObjCurrent)

def _ParsePyObjStatic(obj, Attrs, ObjRoot, ObjCurrent="ObjRoot."):
    if hasattr(obj, "__DollarPath__"):
        ObjCurrent = getattr(obj, "__DollarPath__")
    if isinstance(obj, list):
        ObjParsed = []
        for Index, Item in enumerate(obj):
            ObjParsed.append(_ParsePyObjStatic(Item, [*Attrs, Index], ObjRoot, ObjCurrent))
    elif isinstance(obj, dict):
        ObjParsed = {}
        for Key, Value in obj.items():
            ObjParsed[Key] = _ParsePyObjStatic(Value, [Attrs, Key], ObjRoot, ObjCurrent)
    elif isinstance(obj, str):
        sentence = obj
        while type(sentence) is str and ("$" in sentence in sentence) and ("&" not in sentence):
            sentence = sentence.replace("$^", "ObjRoot.")
            sentence = sentence.replace("$", "ObjCurrent.")
            try:
                # if sentence in ["ObjRoot.param.model"]:
                #     print("aaa")
                sentence = eval(sentence)
            except Exception:
               utils_torch.AddLog("Exception when running %s"%sentence)
        if isinstance(sentence, str) and sentence.startswith("#"):
            sentence = eval(sentence[1:])
        ObjParsed = sentence
    elif utils_torch.json.IsPyObj(obj):
        ObjParsed = EmptyPyObj()
        for Attr, Value in obj.__dict__.items():
            setattr(ObjParsed, Attr, _ParsePyObjStatic(Value, [*Attrs, Attr], ObjRoot, ObjCurrent))
    else:
        #raise Exception("_ParseJsonObj: Invalid type: %s"%type(obj))
        ObjParsed = obj
    return ObjParsed

_ParsePyObj = _ParsePyObjStatic

def ParseParamJsonObj(JsonObj, overwrite=True):
    PyObj = JsonObj2PyObj(JsonObj)
    for name, obj in utils_torch.ListAttrs(PyObj):
        setattr(obj, "__DollarPath__", "root.%s."%name)
    JsonDictsParsed = ParsePyObjStatic(PyObj)
    for Value in JsonDictsParsed.values():
        Value.pop("__DollarPath__")
    for name, obj in utils_torch.ListAttrs(PyObj):
        delattr(obj, "__DollarPath__")
    return JsonDictsParsed

def ParseParamPyObjStatic(PyObj, overwrite=True):
    for name, obj in utils_torch.ListAttrs(PyObj):
        setattr(obj, "__DollarPath__", "root.%s."%name)
    PyObjParsed = ParsePyObjStatic(PyObj)
    for name, obj in utils_torch.ListAttrs(PyObjParsed):
        delattr(obj, "__DollarPath__")
    return PyObjParsed

def ParseJsonObj(JsonObj): # obj can either be dict or list.
    PyObj = JsonObj2PyObj(JsonObj)
    return ParsePyObjStatic(PyObj)
    
JsonObj2ParsedJsonObj = ParseJsonObj

def parse_param_JsonObj(JsonObj, overwrite=True):
    PyObj = JsonObj2PyObj(JsonObj)
    for name, obj in utils_torch.ListAttrs(PyObj):
        setattr(obj, "__DollarPath__", "root.%s."%name)
    JsonDictsParsed = ParsePyObjStatic(PyObj)
    for Value in JsonDictsParsed.Values():
        Value.pop("__DollarPath__")
    for name, obj in utils_torch.ListAttrs(PyObj):
        delattr(obj, "__DollarPath__")
    return JsonDictsParsed

def parse_param_PyObj(PyObj, overwrite=True):
    for name, obj in utils_torch.ListAttrs(PyObj):
        setattr(obj, "__DollarPath__", "root.%s."%name)
    PyObjParsed = ParsePyObjStatic(PyObj)
    for name, obj in utils_torch.ListAttrs(PyObjParsed):
        delattr(obj, "__DollarPath__")
    return PyObjParsed

def parse_JsonObj(JsonObj): # obj can either be dict or list.
    PyObj = JsonObj2PyObj(JsonObj)
    ParsePyObjStatic(PyObj)
    return PyObj

JsonObj2ParsedJsonObj = parse_JsonObj

def JsonObj2ParsedPyObj(JsonObj):
    return JsonObj2PyObj(JsonObj2ParsedJsonObj(JsonObj))

def ParseRoutings(Routings):
    RoutingsParsed = []
    if not isinstance(Routings, list):
        _ParseRouting(Routings)
    else:
        for Index, Routing in enumerate(Routings):
            RoutingsParsed.append(_ParseRouting(Routing))
    return RoutingsParsed

def _ParseRouting(Routing):
    if not isinstance(Routing, str):
        return Routing
    Routing = re.sub(" ", "", Routing) # remove all spaces
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
    for attr, value in ListAttrsAndValues(param):
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

def Register2PyObj(Obj, PyObj, NameList):
    if isinstance(Obj, list):
        RegisterList2PyObj(Obj, PyObj, NameList)
    # elif isinstance(Obj, dict):
    #     RegisterDict2PyObj(Obj, PyObj, Names)
    else:
        if isinstance(NameList, list):
            setattr(PyObj, NameList[0], Obj)
        elif isinstance(NameList, str):
            setattr(PyObj, NameList, Obj)
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
    SetAttrs(Router, "Routings", value=ParseRoutings(Router.Routings))
    RouterParsed = utils_torch.parse.ParsePyObjDynamic(Router, ObjRefList=ObjRefList, RaiseFailedParse=True, **kw)
    return RouterParsed

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