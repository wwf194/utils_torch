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
            # if Arg in ["&~NotifyEpochBatchList"]:
            #     print("AAA")
            #ArgParsed = ResolveArg, **utils_torch.json.PyObj2JsonObj(ContextInfo))
            return ResolveStrDict(Arg, ContextInfo)
        else:
            try:
                return eval(Arg)
            except Exception:
                return Arg
    else:
        return Arg

def ResolveStr(param, **kw):
    kw.setdefault("ObjRoot", utils_torch.GetGlobalParam())
    return ResolveStrDict(param, kw)

def ResolveStrDict(param, ContextInfo):
    if not isinstance(param, str):
        return param
    if "#" in param:
        _param = param
        success = True
        if param.startswith("#"):
            param = param[1:]
        elif param.startswith("&#") or param.startswith("$#"):
            param = param[2:]
        else:
            success = False
        if success:
            param = eval(param)
            if not isinstance(param, str):
                return param
        else:
            param = _param
    if "&" in param:
        # if param in ["&~cache.In.SaveDir"]:
        #     print("aaa")
        ObjRoot = ContextInfo.get("ObjRoot")
        ObjCurrent = ContextInfo.get("ObjCurrent")
        if ContextInfo.get("ObjRefList") is not None:
            ObjRefList = ContextInfo["ObjRefList"]
            sentence = param
            sentence = sentence.replace("&^", "ObjRoot.")
            sentence = sentence.replace("&~", "ObjCurrent.cache.__ParentRef__.")
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
            sentence = sentence.replace("&~", "ObjCurrent.cache.__ParentRef__.")
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
    ObjRefList = kw.get("ObjRefList")
    if ObjRefList is not None:
        # Traverse Attributes in Obj, recursively
        # If an Attribute Value is str and begins with &, redirect this Attribute to Attribute in PyObjRef it points to.
        ObjRefList = utils_torch.ToList(ObjRefList)        
        # if not isinstance(ObjRefList, list):
        #     if hasattr(ObjRefList, "__dict__"):
        #         ObjRefList = [ObjRefList]
        #     else:
        #         raise Exception()
        if InPlace:
            _ParsePyObjDynamicMultiRefsInPlace(Obj, None, None, RaiseFailedParse=RaiseFailedParse, **kw)
            return Obj
        else:
            ObjParsed =  _ParsePyObjDynamicMultiRefs(Obj, None, None, RaiseFailedParse=RaiseFailedParse, **kw)
            return ObjParsed
    else:
        if InPlace: # In place operation
            _ParsePyObjDynamicInPlace(Obj, None, [], RaiseFailedParse, **kw)
            return Obj
        else: # Out of place operation
            kw.setdefault("ExceptionAttrs", [])
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
                if _Attr in kw.get("ExceptionAttrs"):
                    continue
                setattr(ObjParsed, _Attr, _ParsePyObjDynamic(Value, Obj, _Attr, RaiseFailedParse, **kw))
                if _Attr in ["__value__"] and utils_torch.IsDictLikePyObj(ObjParsed):
                    Obj.FromPyObj(ObjParsed)
                    delattr(Obj, _Attr)
                    break
            Sig = False
    elif isinstance(Obj, str) and "&#" in Obj:
        ObjParsed, success = ParseStrWithWell(Obj, Dynamic=True, **kw)
        if success:
            ObjParsed = _ParsePyObjDynamic(ObjParsed, Obj, "(eval)", RaiseFailedParse, **kw)
    elif isinstance(Obj, str) and "&" in Obj:
        # Some Tricks
        if "|-->" in Obj or "/&" in Obj:
            ObjParsed = Obj
            return ObjParsed
        ObjParsed, success = ParseStr(Obj, Dynamic=True, parent=parent, **kw)
        if not success:
            if RaiseFailedParse:
                raise Exception("_ParsePyObjDynamic: Failed to run: %s"%Obj)
            else:
                if kw.get("Verbose"):
                    utils_torch.AddWarning("_ParsePyObjDynamic: Failed to run: %s"%Obj)
        else:
            ObjParsed = _ParsePyObjDynamic(ObjParsed, Obj, "(eval)", RaiseFailedParse, **kw)
    else:
        ObjParsed = Obj
    return ObjParsed

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
        parent = kw.get("parent")
        sentence = Obj
        while "&" in sentence:
            sentence = sentence.replace("&^", "ObjRoot.")
            sentence = sentence.replace("&~", "ObjCurrent.cache.__ParentRef__.")
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

def _ParsePyObjDynamicMultiRefs(Obj, parent, Attr, RaiseFailedParse, **kw):
    # Not In Place. Returns a new ObjParsed.
    # if Obj in ["&^object.agent.cache.Dynamics.Test"]:
    #     print("aaa")
    ObjRefList = kw["ObjRefList"]
    if isinstance(Obj, dict):
        ObjParsed = {}
        for Key, Value in Obj.items():
            ObjParsed[Key] = _ParsePyObjDynamicMultiRefs(Value, parent, Key, RaiseFailedParse, **kw)
    elif isinstance(Obj, list):
        ObjParsed = []
        for Index, Item in enumerate(Obj):
            ObjParsed.append(_ParsePyObjDynamicMultiRefs(Item, parent, Index, RaiseFailedParse, **kw))
    elif isinstance(Obj, utils_torch.PyObj):
        if Obj.IsResolveBase():
            kw["ObjCurrent"] = Obj
        ObjParsed = utils_torch.EmptyPyObj()
        for Attr, Value in ListAttrsAndValues(Obj, ExcludeCache=False):
            setattr(ObjParsed, Attr, _ParsePyObjDynamicMultiRefs(getattr(Obj, Attr), parent, Attr, RaiseFailedParse, **kw))
    elif isinstance(Obj, str):
        # if Obj in ["#utils_torch.GetMainSaveDir"]:
        #     print("aaa")
        if "#" in Obj:
            ObjParsed, success = ParseStrWithWell(Obj, Dynamic=True, **kw)
            if success:
                ObjParsed = _ParsePyObjDynamicMultiRefs(ObjParsed, Obj, "(eval)", RaiseFailedParse, **kw)
                return ObjParsed
            else:
                pass # don't return
        if "&" in Obj:
            success = False
            for ObjRef in ObjRefList:
                try:
                    kw["ObjCurrent"] = ObjRef
                    ObjParsed, success = ParseStr(
                        Obj, Dynamic=True, parent=parent, Verbose=False, **kw
                    )
                    if success:
                        break
                except Exception:
                    pass
                    #utils_torch.AddLog("Failed to resoolve to current PyObjRef. Try redirecting to next PyObjRef.")
            if not success:
                report = "_ParsePyObjDynamicMultiRefs: Failed to resolve to any PyObjRef in given ObjRefList by running: %s"%Obj
                if RaiseFailedParse:
                    raise Exception(report)
                else:
                    utils_torch.AddWarning(report)
                    ObjParsed = Obj
            else:
                return ObjParsed
        ObjParsed = Obj
    else:
        ObjParsed = Obj
    return ObjParsed

def _ParsePyObjDynamicMultiRefsInPlace(Obj, parent, Attr, RaiseFailedParse, **kw):
    ObjRefList = kw["ObjRefList"]
    if isinstance(Obj, dict):
        for Key, Value in Obj.items():
            _ParsePyObjDynamicMultiRefsInPlace(Value, Obj, Key, RaiseFailedParse, **kw)
    elif isinstance(Obj, list):
        for Index, Item in enumerate(Obj):
            _ParsePyObjDynamicMultiRefsInPlace(Item, Obj, Index, RaiseFailedParse, **kw)
    elif isinstance(Obj, utils_torch.PyObj):
        if Obj.IsResolveBase():
            kw["ObjCurrent"] = Obj
        ObjParsed = utils_torch.PyObj()
        for Attr, Value in ListAttrsAndValues(Obj):
            _ParsePyObjDynamicMultiRefsInPlace(Value, Obj, Attr, RaiseFailedParse, **kw)
    elif isinstance(Obj, str):
        ObjRoot = kw.get("ObjRoot")
        ObjRef = kw.get("ObjRef")
        sentence = Obj
        if "&" in sentence:
            success = False
            sentence = sentence.replace("&^", "ObjRoot.")
            sentence = sentence.replace("&~", "ObjCurrent.cache.__ParentRef__.")
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
                report = "_ParsePyObjDynamicMultiRefsInPlace: Failed to resolve to any PyObjRef in given ObjRefList by running: %s"%sentence
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

def ApplyMethodOnPyObj(Obj, Function=lambda x:(x, True), **kw):
    # Not Inplace
    return _ApplyMethodOnPyObj(Obj, Function, [], **kw)

def _ApplyMethodOnPyObj(Obj, Function, Attrs, **kw):
    Obj, ContinueParse = Function(Obj)
    if not ContinueParse:
        return Obj
    if isinstance(Obj, list) or utils_torch.IsListLikePyObj(Obj):
        ObjParsed = []
        for Index, Item in enumerate(Obj):
            ObjParsed.append(_ApplyMethodOnPyObj(Item, Function, [*Attrs, Index], **kw))
    elif isinstance(Obj, dict):
        ObjParsed = {}
        for Key, Value in Obj.items():
            ObjParsed[Key] = _ApplyMethodOnPyObj(Value, Function, [*Attrs, Key], **kw)
    elif utils_torch.IsDictLikePyObj(Obj):
        ObjParsed = utils_torch.EmptyPyObj()
        for Attr, Value in ListAttrsAndValues(Obj):
            setattr(ObjParsed, Attr, _ApplyMethodOnPyObj(Value, Function, [*Attrs, Attr], **kw))
    else:
        ObjParsed = Obj
    return ObjParsed

def ParsePyObjStatic(Obj, **kw):
    kw.setdefault("WithinJson", True)
    kw.setdefault("RecurDepth", 1)
    InPlace = kw.setdefault("InPlace", True)
    ObjCurrent = kw.setdefault("ObjCurrent", Obj)
    ObjRoot = kw.setdefault("ObjRoot", utils_torch.GetGlobalParam())
    _ParseResolveBaseInPlace(Obj, None, None, ParsedObj=defaultdict(lambda:None), **kw)
    if InPlace:
        _ParsePyObjStaticInPlace(
            Obj, None, None, 
            **kw
            #ParsedObj=defaultdict(lambda:None)
        )
        return Obj
    else:
        return _ParsePyObjStatic(Obj, None, None, ParsedObj=defaultdict(lambda:None), **kw)

def _ParseResolveBaseInPlace(Obj, parent, Attr, WithinJson=True, **kw):
    kw.setdefault("RecurDepth", 1)
    kw["RecurDepth"] += 1

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
            _ParseResolveBaseInPlace(Item, Obj, Index, WithinJson=WithinJson, **kw)            
    elif isinstance(Obj, dict):
        for Key, Value in Obj.items():
            _ParseResolveBaseInPlace(Value, Obj, Key, WithinJson=WithinJson, **kw)
    elif utils_torch.IsPyObj(Obj):
        if Obj.IsResolveBase():
            setattr(Obj.cache, "__ParentRef__", kw.get("ObjCurrent"))
            kw["ObjCurrent"] = Obj
        setattr(Obj.cache, "__ResolveRef__", kw.get("ObjCurrent"))
        for _Attr, Value in ListAttrsAndValues(Obj):
            _ParseResolveBaseInPlace(Value, Obj, _Attr, WithinJson=WithinJson, **kw)
    else:
        pass

def _ParsePyObjStaticInPlace(Obj, parent, Attr, **kw):
    # if Obj in ["$Batch.Size"]:
    #     print("aaa")
    kw["RecurDepth"] += 1
    WithinJson = kw["WithinJson"]
    if isinstance(Obj, list):
        for Index, Item in enumerate(Obj):
            _ParsePyObjStaticInPlace(Item, Obj, Index, **kw)
    elif isinstance(Obj, dict):
        Obj = utils_torch.PyObj(Obj)
        parent[Attr] = Obj
        for Key, Value in Obj.Items():
            _ParsePyObjStaticInPlace(Value, Obj, Key, **kw)
    elif utils_torch.IsPyObj(Obj):
        if Obj.IsResolveBase():
            kw["ObjCurrent"] = Obj   
        Sig = True
        while Sig:
            for _Attr, Value in ListAttrsAndValues(Obj):
                _ParsePyObjStaticInPlace(Value, Obj, _Attr, **kw)
                if _Attr in ["__value__"] and isinstance(getattr(Obj, _Attr), utils_torch.PyObj):
                    Obj.FromPyObj(getattr(Obj, _Attr))
                    delattr(Obj, "__value__")
                    break
            Sig = False
    elif isinstance(Obj, str):
        if type(Obj) is str and ("$" in Obj) and ("&" not in Obj):
            Obj = Obj.lstrip("#")
            while type(Obj) is str and ("$" in Obj) and ("&" not in Obj):
                Obj, success = ParseStr(Obj, Dynamic=False, parent=parent, **kw)
                if success:
                    continue
                Obj, success = ParseStrLocal(Obj, Dynamic=False, **kw)
                if success:
                    continue
                break
            parent[Attr] = Obj
        elif Obj.startswith("#"):
            _Obj = Obj
            try:
                sentence = utils_torch.RemoveHeadTailWhiteChars(Obj.lstrip("#"))
                Obj = eval(sentence)
            except Exception:
                Obj = _Obj
            if not IsJsonObj(Obj) and WithinJson:
                utils_torch.AddLog("_ParsePyObjStaticInPlace: Not a Json Obj: %s of type %s ."%(Obj, type(Obj)))
            else:
                parent[Attr] = Obj
        else:
            pass
    else:
        pass

def ParseStr(Str, Dynamic=False, Verbose=True, **kw):
    # if Str in ["&^object.agent.Dynamics.InitBeforeEpochTrain"]:
    #     print("aaa")

    Str = utils_torch.RemoveHeadTailWhiteChars(Str)
    ObjCurrent = kw.get("ObjCurrent")
    ObjRoot = kw.get("ObjRoot")
    parent = kw.get("parent")

    _Str = Str
    sentence = Str
    if Dynamic:
        while "&" in sentence:
            sentence = sentence.replace("&^", "ObjRoot.")
            sentence = sentence.replace("&~", "ObjCurrent.cache.__ParentRef__.")
            sentence = sentence.replace("&*", "ObjCurrent.cache.__object__.")
            sentence = sentence.replace("&", "ObjCurrent.")
    else:
        while "$" in sentence:
            sentence = sentence.replace("$^", "ObjRoot.")
            sentence = sentence.replace("$~", "parent.")
            sentence = sentence.replace("$*", "ObjCurrent.cache.__object__.")
            sentence = sentence.replace("$", "ObjCurrent.")
        
    success = False
    try:
        Str = eval(sentence)
        if Str in ["__ToBeSet__"]:
            success = False
            raise Exception()
        else:
            success = True
    except Exception:
        Str = _Str
        success = False
        # if Verbose:
        #     utils_torch.AddLog("ParseStr: Failed to run %s"%sentence)
    if success:
        if not utils_torch.IsJsonObj(Str):
            if kw.get("WithinJson"):
                # if Verbose:
                #     utils_torch.AddLog("ParseStr: Not a Json Obj: %s of type %s ."%(Str, type(Str)))
                success = False
                Str = _Str
    return Str, success

def ParseStrWithWell(Str, Dynamic=True, **kw):
    Str = utils_torch.RemoveHeadTailWhiteChars(Str)
    _Str = Str
    if Dynamic:
        if Str.startswith("&#"):
            Str = Str[2:]
        else:
            return _Str, False
    else:
        if Str.startswith("$#"):
            Str = Str[2:]
        elif Str.startswith("#"):
            Str = Str[1:]
        else:
            return _Str, False
    try:
        ObjParsed = eval(Str)
        success = True
    except Exception:
        success = False
        if kw.get("RaiseFailedParse"):
            raise Exception("ParseStrWithWell: Failed to run: %s"%Str)
        else:
            # if Str in ["utils_torch.GetMainSaveDir"]:
            #     print("aaa")
            utils_torch.AddWarning("ParseStrWithWell: Failed to run: %s"%Str)
            ObjParsed = _Str
    return ObjParsed, success

def ParseStrLocal(Str, Dynamic=False, **kw):
    _Str = Str
    ObjCurrent = kw.get("ObjCurrent")
    ObjRoot = kw.get("ObjRoot")
    parent = kw.get("parent")
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
            utils_torch.AddLog("_ParsePyObjStaticInPlace: Not a Json Obj: %s of type %s ."%(Str, type(Str)))
            success = False

    return Str, success 

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
        ObjParsed = utils_torch.PyObj()
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


def ParseParamStaticAndDynamic(Args):
    ParseParamStatic(Args)
    ParseParamDynamic(Args)
    return

def ParseParamStatic(Args, Save=False, SavePath=None):
    if SavePath is None:
        SavePath = utils_torch.GetMainSaveDir() + "param_parsed_static.jsonc"
    GlobalParam = utils_torch.GetGlobalParam()
    param = GlobalParam.param
    utils_torch.json.PyObj2JsonFile(param, SavePath)
    utils_torch.parse.ParsePyObjStatic(param, ObjCurrent=param, ObjRoot=utils_torch.GetGlobalParam(), InPlace=True)
    if Save:
        SavePath = utils_torch.RenameIfFileExists(SavePath)
        utils_torch.json.PyObj2JsonFile(param, SavePath)
    return

def ParseParamDynamic(Args, Save=False, SavePath=None):
    GlobalParam = utils_torch.GetGlobalParam()
    if SavePath is None:
        utils_torch.GetMainSaveDir() + "param_parsed_dynamic.jsonc"
    for attr, param in utils_torch.ListAttrsAndValues(GlobalParam.param):
        utils_torch.parse.ParsePyObjDynamic(param, ObjCurrent=param, ObjRoot=utils_torch.GetGlobalParam(), InPlace=True)
    if Save:
        utils_torch.json.PyObj2JsonFile(GlobalParam.param, utils_torch.RenameIfFileExists(SavePath))
    return

def ParseClass(ClassPath):
    try:
        Module = utils_torch.ImportModule(ClassPath)
        if hasattr(Module, "__MainClass__"):
            return Module.__MainClass__
        else:
            return Module
    except Exception:
        Class = eval(ClassPath)
        return Class
