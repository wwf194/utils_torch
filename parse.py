from sre_constants import SUCCESS
from utils_torch.attrs import *
from utils_torch.json import *

from collections import defaultdict

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

def ParseFunctionArgs(Args, ContextInfo):
    # if type(Args) is not list:
    #     raise Exception(Args)
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
            return ContextInfo.get(Arg)
        elif "&" in Arg:
            #ArgParsed = ResolveArg, **utils_torch.json.PyObj2JsonObj(ContextInfo))
            return ResolveStrDict(Arg, ContextInfo)
        else:
            return Arg
    else:
        return Arg

def ResolveStr(param, **kw):
    return ResolveStrDict(param, kw)

def ResolveStrDict(param, ContextInfo):
    if not isinstance(param, str):
        return param
    if "&" in param:
        ObjRoot = ContextInfo.get("ObjRoot")
        ObjCurrent = ContextInfo.get("ObjCurrent")
        if ContextInfo.get("ObjRefList") is not None:
            ObjRefList = ContextInfo["ObjRefList"]
            sentence = param
            sentence = sentence.replace("&^", "ObjRoot.")
            sentence = sentence.replace("&~", "ObjCurrent.parent")
            sentence = sentence.replace("&*", "ObjCurrent.cache.__object__.")
            sentence = sentence.replace("&", "ObjRef.")
            for ObjRef in ObjRefList:
                try:
                    result = eval(sentence)
                    return result
                except Exception:
                    continue
            utils_torch.AddWarning("Failed to resolve to any in ObjRefList by running: %s"%sentence)
        else:
            sentence = param
            sentence = sentence.replace("&^", "ObjRoot.")
            sentence = sentence.replace("&*", "ObjCurrent.cache.__object__.")
            sentence = sentence.replace("&", "ObjCurrent.")
        try:
            return eval(sentence)
        except Exception:
            utils_torch.AddWarning("ResolveStrDict: Failed to run: %s"%sentence)
            return param
    else:
        return eval(param)

def ParsePyObjDynamic(Obj, RaiseFailedParse=False, InPlace=False, **kw):
    #utils_torch.json.CheckIsPyObj(Obj)
    if kw.get("ObjRefList") is not None:
        return ParsePyObjDynamicWithMultiRefs(
            Obj, 
            RaiseFailedParse=RaiseFailedParse, 
            InPlace=InPlace, 
            **kw
        )
    else:
        if InPlace:
            _ParsePyObjDynamicInPlace(Obj, None, [], RaiseFailedParse, **kw)
            return Obj
        else:
            return _ParsePyObjDynamic(Obj, None, [], RaiseFailedParse, **kw)

def _ParsePyObjDynamic(Obj, parent, attr, RaiseFailedParse, **kw):
    if isinstance(Obj, list):
        ObjParsed = []
        for Index, Item in enumerate(Obj):
            ObjParsed.append(_ParsePyObjDynamic(Item, Obj, Index, RaiseFailedParse, **kw))
    elif isinstance(Obj, dict):
        ObjParsed = {}
        for Key, Value in Obj.items():
            ObjParsed[Key] = _ParsePyObjDynamic(Value, Obj, Key, RaiseFailedParse, **kw)
    elif isinstance(Obj, utils_torch.PyObj):
        if Obj.IsResolveBase():
            kw["ObjCurrent"] = Obj
        ObjParsed = utils_torch.EmptyPyObj()
        Sig = True
        while Sig:
            for _Attr, Value in ListAttrsAndValues(Obj, ExcludeCache=False):
                setattr(ObjParsed, _Attr, _ParsePyObjDynamic(Value, Obj, _Attr, RaiseFailedParse, **kw))
                if _Attr in ["__value__"] and utils_torch.IsDictLikePyObj(ObjParsed):
                    Obj.FromPyObj(ObjParsed)
                    delattr(Obj, _Attr)
                    break
            Sig = False
    elif isinstance(Obj, str) and "&" in Obj:
        # Some Tricks
        if "|-->" in Obj:
            ObjParsed = Obj
            return ObjParsed
        success, ObjParsed = ParseStrDynamic(Obj, **kw)
        if not success:
            if RaiseFailedParse:
                raise Exception("_ParsePyObjDynamic: Failed to run: %s"%Obj)
            else:
                utils_torch.AddWarning("_ParsePyObjDynamic: Failed to run: %s"%Obj)
                return ObjParsed
        else:
            ObjParsed = _ParsePyObjDynamic(ObjParsed, Obj, "(eval)", RaiseFailedParse, **kw)
    else:
        ObjParsed = Obj
    return ObjParsed

def ParseStrDynamic(Str, **kw):
    success = True
    _Str = Str
    ObjRoot= kw.get("ObjRoot")
    ObjCurrent = kw.get("ObjCurrent")

    sentence = Str
    sentence = sentence.replace("&^", "ObjRoot.")
    sentence = sentence.replace("&~", "parent.")
    sentence = sentence.replace("&*", "ObjCurrent.cache.__object__.")
    sentence = sentence.replace("&", "ObjCurrent.")

    try:
        Str = eval(sentence)
    except Exception:
        success = False
        Str = _Str
    return success, Str

def _ParsePyObjDynamicInPlace(Obj, parent, attr, RaiseFailedParse, **kw):
    if isinstance(Obj, list):
        for Index, Item in enumerate(Obj):
            _ParsePyObjDynamicInPlace(Item, Obj, Index, RaiseFailedParse, **kw)
    elif isinstance(Obj, dict):
        for Key, Value in Obj.items():
            _ParsePyObjDynamicInPlace(Value, Obj, Key, RaiseFailedParse, **kw)
    elif isinstance(Obj, utils_torch.PyObj):
        if hasattr(Obj, "__ResolveBase__"):
            kw["ObjCurrent"] = Obj
        Sig = True
        while Sig:
            for _Attr, Value in Obj.__dict__.items():
                _ParsePyObjDynamicInPlace(Value, Obj, _Attr, RaiseFailedParse, **kw)
                if _Attr in ["__value__"] and utils_torch.IsDictLikePyObj(getattr(Obj, _Attr)):
                    Obj.FromPyObj(getattr(Obj, _Attr))
                    delattr(Obj, "__value__")
                    break
            Sig = False
    elif isinstance(Obj, str) and "&" in Obj:
        success = False
        ObjRoot= kw.get("ObjRoot")
        ObjCurrent = kw.get("ObjCurrent")
        sentence = Obj
        sentence = sentence.replace("&^", "ObjRoot.")
        sentence = sentence.replace("&~", "parent.")
        sentence = sentence.replace("&", "ObjCurrent.")
        
        # Some Tricks
        if "|-->" in sentence:
            return
 
        if RaiseFailedParse:
            ObjParsed = eval(sentence)
        else: 
            try:
                ObjParsed = eval(sentence)
                success = True
            except Exception:
                # if sentence in ["ObjRoot.Object.world.GetArenaByIndex(0).BoundaryBox.Size * 0.07"]:
                #     print("aaa")
                utils_torch.AddWarning("_ParsePyObjDynamicInPlace: Failed to run: %s"%sentence)
                return
        parent[attr] = ObjParsed
    else:
        pass

def ParsePyObjDynamicWithMultiRefs(Obj, RaiseFailedParse=False, InPlace=False, **kw):
    ObjRefList = kw["ObjRefList"]
    # Traverse Attributes in Obj, recursively
    # If an Attribute Value is str and begins with &, redirect this Attribute to Attribute in PyObjRef it points to.
    if not isinstance(ObjRefList, list):
        if hasattr(ObjRefList, "__dict__"):
            ObjRefList = [ObjRefList]
        else:
            raise Exception()
    kw["ObjRefList"] = ObjRefList
    if InPlace:
        _ParsePyObjDynamicWithMultiRefsInPlace(Obj, None, None, RaiseFailedParse=RaiseFailedParse, **kw)
        return Obj
    else:
        return _ParsePyObjDynamicWithMultiRefs(Obj, None, None, RaiseFailedParse=RaiseFailedParse, **kw)

def _ParsePyObjDynamicWithMultiRefs(Obj, parent, Attr, RaiseFailedParse, **kw):
    ObjRefList = kw["ObjRefList"]
    if isinstance(Obj, dict):
        ObjParsed = {}
        for Key, Value in Obj.items():
            ObjParsed[Key] = _ParsePyObjDynamicWithMultiRefs(Value, parent, Key, RaiseFailedParse, **kw)
    elif isinstance(Obj, list):
        ObjParsed = []
        for Index, Item in enumerate(Obj):
            ObjParsed.append(_ParsePyObjDynamicWithMultiRefs(Item, parent, Index, RaiseFailedParse, **kw))
    elif isinstance(Obj, utils_torch.PyObj):
        if Obj.IsResolveBase():
            kw["ObjCurrent"] = Obj
        ObjParsed = utils_torch.EmptyPyObj()
        for Attr, Value in ListAttrsAndValues(Obj, ExcludeCache=False):
            setattr(ObjParsed, Attr, _ParsePyObjDynamicWithMultiRefs(getattr(Obj, Attr), parent, Attr, RaiseFailedParse, **kw))
    elif isinstance(Obj, str):
        # if Obj in ["&GetBias"]:
        #     print("aaa")
        ObjRoot = kw.get("ObjRoot")
        ObjRef = kw.get("ObjRef")
        sentence = Obj
        # if "CellStateDecay" in sentence:
        #     print("aaa")
        if "&" in sentence:
            success = False
            sentence = sentence.replace("&^", "ObjRoot.")
            sentence = sentence.replace("&~", "parent.")
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
                report = "_ParsePyObjDynamicWithMultiRefs: Failed to resolve to any PyObjRef in given ObjRefList by running: %s"%sentence
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

def _ParsePyObjDynamicWithMultiRefsInPlace(Obj, parent, Attr, RaiseFailedParse, **kw):
    ObjRefList = kw["ObjRefList"]
    if isinstance(Obj, dict):
        for Key, Value in Obj.items():
            _ParsePyObjDynamicWithMultiRefsInPlace(Value, Obj, Key, RaiseFailedParse, **kw)
    elif isinstance(Obj, list):
        for Index, Item in enumerate(Obj):
            _ParsePyObjDynamicWithMultiRefsInPlace(Item, Obj, Index, RaiseFailedParse, **kw)
    elif isinstance(Obj, utils_torch.PyObj):
        if Obj.IsResolveBase():
            kw["ObjCurrent"] = Obj
        ObjParsed = utils_torch.PyObj()
        for Attr, Value in ListAttrsAndValues(Obj):
            _ParsePyObjDynamicWithMultiRefsInPlace(Value, Obj, Attr, RaiseFailedParse, **kw)
    elif isinstance(Obj, str):
        ObjRoot = kw.get("ObjRoot")
        ObjRef = kw.get("ObjRef")
        sentence = Obj
        if "&" in sentence:
            success = False
            sentence = sentence.replace("&^", "ObjRoot.")
            sentence = sentence.replace("&~", "parent.")
            sentence = sentence.replace("&*", "ObjRef.cache.__object__.")
            sentence = sentence.replace("&", "ObjRef.")
            for ObjRef in ObjRefList:
                try:
                    ObjParsed = eval(sentence)
                    success = True
                    break
                except Exception:
                    pass
                    #utils_torch.AddLog("Failed to resolve to current PyObjRef. Try redirecting to next PyObjRef.")
            if not success:
                report = "_ParsePyObjDynamicWithMultiRefsInPlace: Failed to resolve to any PyObjRef in given ObjRefList by running: %s"%sentence
                if RaiseFailedParse:
                    raise Exception(report)
                else:
                    utils_torch.AddWarning(report)
                    return
            else:
                parent[Attr] = ObjParsed
        else:
            ObjParsed = sentence
    else:
        pass

def ProcessPyObj(Obj, Function=lambda x:(x, True), **kw):
    # Not Inplace
    ObjParsed = utils_torch.PyObj()
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
        ObjParsed = utils_torch.EmptyPyObj()
        for Attr, Value in Obj.__dict__.items():
            setattr(ObjParsed, Attr, _ParsePyObj(Value, Obj, [*Attrs, Attr]))
    else:
        ObjParsed = Obj
    return ObjParsed

def ParsePyObjStatic(Obj, ObjRoot=None, ObjCurrent=None, InPlace=True, **kw):
    #CheckIsPyObj(Obj)
    _ParseResolveBaseInPlace(Obj, None, None, ObjRoot=ObjRoot, ObjCurrent=ObjCurrent, ParsedObj=defaultdict(lambda:None))
    if InPlace:
        _ParsePyObjStaticInPlace(Obj, None, None, ObjRoot=ObjRoot, ObjCurrent=ObjCurrent, ParsedObj=defaultdict(lambda:None))
        return Obj
    else:
        return _ParsePyObjStatic(Obj, None, None, ObjRoot=ObjRoot, ObjCurrent=ObjCurrent, ParsedObj=defaultdict(lambda:None))

def _ParseResolveBaseInPlace(Obj, parent, Attr, RemainJson=True, **kw):
    kw.setdefault("RecurDepth", 1)
    kw["RecurDepth"] += 1
    if kw["RecurDepth"] > 200:
        print("aaa")
    # if kw["ParsedObj"][id(Obj)] is not None:
    #     return
    # kw["ParsedObj"][id(Obj)] = "Parsed"
    kw.setdefault("Attrs", [])
    kw["Attrs"].append(Attr)
    Attrs = kw["Attrs"]
    if isinstance(Obj, list):
        if parent is not None and Attr not in ["__value__"]:
            SetAttr(parent, Attr, utils_torch.PyObj({
                "__value__": Obj,
            }))
            Obj = GetAttr(parent, Attr)
            setattr(Obj.cache, "__ResolveRef__", parent)
            Obj = Obj.__value__
        for Index, Item in enumerate(Obj):
            _ParseResolveBaseInPlace(Item, Obj, Index, RemainJson, **kw)            
    elif isinstance(Obj, dict):
        for Key, Value in Obj.items():
            _ParseResolveBaseInPlace(Value, Obj, Key, RemainJson, **kw)
    elif utils_torch.IsPyObj(Obj):
        if Obj.IsResolveBase():
            kw["ObjCurrent"] = Obj
        setattr(Obj.cache, "__ResolveRef__", kw.get("ObjCurrent"))
        for _Attr, Value in ListAttrsAndValues(Obj):
            # if _Attr in ["DynamicParseAttrs"]:
            #     print("aaa")
            # print(_Attr)
            # print(Value)
            _ParseResolveBaseInPlace(Value, Obj, _Attr, RemainJson, **kw)
    else:
        pass

def _ParsePyObjStaticInPlace(Obj, parent, Attr, RemainJson=True, **kw):
    kw.setdefault("RecurDepth", 1)
    kw["RecurDepth"] += 1
    if kw["RecurDepth"] > 200:
        print("aaa")
    # if kw["ParsedObj"][id(Obj)] is not None:
    #     return
    # kw["ParsedObj"][id(Obj)] = "Parsed"
    # if Obj in ["$Loss.ActivityConstrain.Coefficient"]:
    #     print("aaa")
    if isinstance(Obj, list):
        for Index, Item in enumerate(Obj):
            _ParsePyObjStaticInPlace(Item, Obj, Index, RemainJson, **kw)
    elif isinstance(Obj, dict):
        for Key, Value in Obj.items():
            _ParsePyObjStaticInPlace(Value, Obj, Key, RemainJson, **kw)
    elif utils_torch.IsPyObj(Obj):
        if Obj.IsResolveBase():
            kw["ObjCurrent"] = Obj   
        Sig = True
        while Sig:
            for _Attr, Value in ListAttrsAndValues(Obj):
                _ParsePyObjStaticInPlace(Value, Obj, _Attr, RemainJson, **kw)
                if _Attr in ["__value__"] and isinstance(getattr(Obj, _Attr), utils_torch.PyObj):
                    Obj.FromPyObj(getattr(Obj, _Attr))
                    delattr(Obj, "__value__")
                    break
            Sig = False
    elif isinstance(Obj, str):
        RecurLimit = 20
        if type(Obj) is str and ("$" in Obj) and ("&" not in Obj) and RecurLimit:
            Obj = Obj.lstrip("#")
            while type(Obj) is str and ("$" in Obj) and ("&" not in Obj) and RecurLimit > 0:
                success, Obj = ParseStrStatic(Obj, parent, **kw)
                if success:
                    continue

                success, Obj = ParseStr2Static(Obj, parent, **kw)
                if success:
                    continue

                RecurLimit -= 1
                break     
            
            parent[Attr] = Obj
        elif Obj.startswith("#"):
            _Obj = Obj
            try:
                sentence = utils_torch.RemoveHeadTailWhiteChars(Obj.lstrip("#"))
                Obj = eval(sentence)
            except Exception:
                Obj = _Obj

            if not IsJsonObj(Obj):
                utils_torch.AddLog("_ParsePyObjStaticInPlace: Not a Json Obj: %s of type %s ."%(Obj, type(Obj)))
                Obj = _Obj

            parent[Attr] = Obj
        else:
            pass
    else:
        pass

def ParseStrStatic(Str, parent, **kw):
    _Str = Str
    sentence = Str
    ObjCurrent = kw.get("ObjCurrent")
    ObjRoot = kw.get("ObjRoot")

    sentence = sentence.replace("$^", "ObjRoot.")
    sentence = sentence.replace("$~", "parent.")
    sentence = sentence.replace("$*", "ObjCurrent.cache.__object__.")
    sentence = sentence.replace("$", "ObjCurrent.")
    success = False

    try:
        # if sentence in ["ObjRoot.param.model"]:
        #     print("aaa")
        Str = eval(sentence)
        if Str in ["__ToBeSet__"]:
            success = False
            raise Exception()
        else:
            success = True
    except Exception:
        Str = _Str
        success = False
        utils_torch.AddLog("_ParsePyObjStaticInPlace: Failed to run %s"%sentence)

    if success:
        if not utils_torch.IsJsonObj(Str):
            Str = _Str
            utils_torch.AddLog("_ParsePyObjStaticInPlace: Not a Json Obj: %s of type %s ."%(Str, type(Str)))
            success = False
        else:
            success = True
    
    return success, Str

def ParseStr2Static(Str, parent, **kw):
    _Str = Str
    ObjCurrent = kw.get("ObjCurrent")
    ObjRoot = kw.get("ObjRoot")
    # if Str in ["$^param.agent.HiddenNeurons.Num.($^param.agent.Task)"]:
    #     print("aaa")
    # if "data: [data[" in Str:
    #     print("aaaa")
    success = True
    MatchResult = re.match(r"^(.*)(\(\$[^\)]*\))(.*)$", Str)
    if MatchResult is None:
        success = False
        Str = _Str

    if success:
        sentence = MatchResult.group(2)
        sentence = sentence.replace("$^", "ObjRoot.")
        sentence = sentence.replace("$~", "parent.")
        sentence = sentence.replace("$*", "ObjCurrent.cache.__object__.")
        sentence = sentence.replace("$", "ObjCurrent.")

        try:
            Str = eval(sentence)
            if isinstance(Str, str) and "$" in Str:
                Str = "(" + Str + ")"
            if Str in ["__ToBeSet__"]:
                raise Exception()
            Str = MatchResult.group(1) + str(Str) + MatchResult.group(3)
        except Exception:
            Str = _Str
            success = False

    if success:
        if not IsJsonObj(Str):
            Str = _Str
            utils_torch.AddLog("_ParsePyObjStaticInPlace: Not a Json Obj: %s of type %s ."%(Obj, type(Obj)))
            success = False

    return success, Str

def _ParsePyObjStatic(Obj, parent, Attr, **kw):

    if isinstance(Obj, list):
        ObjParsed = []
        for Index, Item in enumerate(Obj):
            ObjParsed.append(_ParsePyObjStatic(Item, Obj, Index, **kw))
    elif isinstance(Obj, dict):
        ObjParsed = {}
        for Key, Value in Obj.items():
            ObjParsed[Key] = _ParsePyObjStatic(Value, Obj, Key, **kw)
    elif utils_torch.IsPyObj(Obj):
        if hasattr(Obj, "__ResolveBase__"):
            kw["ObjCurrent"] = Obj
        ObjParsed = PyObj()
        Sig = True
        while Sig:
            for _Attr, Value in ListAttrsAndValues(Obj, Exceptions=["__ResolveRef__"]):
                setattr(ObjParsed, _Attr, _ParsePyObjStatic(Value, Obj, _Attr, **kw))
                if _Attr in ["__value__"] and isinstance(ObjParsed, utils_torch.PyObj):
                    Obj.FromPyObj(ObjParsed)
                    delattr(Obj, _Attr)
                    break
            Sig = False
    elif isinstance(Obj, str):
        if hasattr(Obj, "__ResolveBase__"):
            ObjCurrent = getattr(Obj, "__ResolveBase__")
        sentence = Obj
        while type(sentence) is str and ("$" in sentence in sentence) and ("&" not in sentence):
            if sentence in ["$Num", "$^param.agent.Modules.model.Neurons.Num // 4"]:
                print("bbb")
            ObjCurrent = kw.get("ObjCurrent")
            ObjRoot = kw.get("ObjRoot")
            sentence = sentence.replace("$^", "ObjRoot.")
            sentence = sentence.replace("$~", "parent.")
            sentence = sentence.replace("$*", "ObjCurrent.cache.__object__.")
            sentence = sentence.replace("$", "ObjCurrent.")
            try:
                # if sentence in ["ObjRoot.param.model"]:
                #     print("aaa")
                sentence = eval(sentence)
            except Exception:
               utils_torch.AddLog("_ParsePyObjStatic: Exception when running %s"%sentence)
        if isinstance(sentence, str) and sentence.startswith("#"):
            sentence = utils_torch.RemoveHeadTailWhiteChars(sentence.lstrip("#"))
            sentence = eval(sentence)
        ObjParsed = sentence
    else:
        #raise Exception("_ParseJsonObj: Invalid type: %s"%type(Obj))
        ObjParsed = Obj
    return ObjParsed

_ParsePyObj = _ParsePyObjStatic

# def ParseParamJsonObj(JsonObj, overwrite=True):
#     PyObj = JsonObj2PyObj(JsonObj)
#     for name, Obj in utils_torch.ListAttrs(PyObj):
#         setattr(Obj, "__ResolveBase__", "root.%s."%name)
#     JsonDictsParsed = ParsePyObjStatic(PyObj)
#     for Value in JsonDictsParsed.values():
#         Value.pop("__ResolveBase__")
#     for name, Obj in utils_torch.ListAttrs(PyObj):
#         delattr(Obj, "__ResolveBase__")
#     return JsonDictsParsed

# def ParseParamPyObjStatic(PyObj, overwrite=True):
#     for name, Obj in utils_torch.ListAttrs(PyObj):
#         setattr(Obj, "__ResolveBase__", "root.%s."%name)
#     PyObjParsed = ParsePyObjStatic(PyObj)
#     for name, Obj in utils_torch.ListAttrs(PyObjParsed):
#         delattr(Obj, "__ResolveBase__")
#     return PyObjParsed

def ParseJsonObj(JsonObj): # Obj can either be dict or list.
    PyObj = JsonObj2PyObj(JsonObj)
    return ParsePyObjStatic(PyObj)
    
JsonObj2ParsedJsonObj = ParseJsonObj

def JsonObj2ParsedPyObj(JsonObj):
    return JsonObj2PyObj(JsonObj2ParsedJsonObj(JsonObj))


def FilterFromPyObj(PyObj, Keys):
    List = []
    for Key in Keys:
        Value = GetAttrs(PyObj, Key)
        List.append(Value)
    return List

def Register2PyObj(Obj, PyObj, NameList):
    if isinstance(NameList, str):
        NameList = [NameList]

    if isinstance(Obj, list):
        ObjNum = len(Obj)
        NameNum = len(NameList)
        if ObjNum==NameNum:
            RegisterList2PyObj(Obj, PyObj, NameList)
        else:
            if NameNum==1:
                setattr(PyObj, NameList[0], Obj)
            else:
                report = "ObjNum: %d NameNum: %d\n"%(ObjNum, NameNum)
                report += "NameList: %s"%NameList
                raise Exception(report)
    elif Obj is None:
        return
    else:
        if len(NameList)==1:
            setattr(PyObj, NameList[0], Obj)
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