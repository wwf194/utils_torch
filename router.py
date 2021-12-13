import re
import utils_torch
from utils_torch.attr import *

def ParseRouterStaticAndDynamic(Router, **kw):
    ParseRouterStatic(Router, InPlace=True)
    RouterParsed = ParseRouterDynamic(Router, **kw)
    return RouterParsed
ParseRouter = ParseRouterStaticAndDynamic

def ParseRoutersDynamic(Routers, ObjRefList=[], **kw):
    if isinstance(Routers, list):
        RouterParsed = []
        for Router in Routers:
            RouterParsed.append(ParseRouterDynamic(Router, ObjRefList, **kw))
        return RouterParsed
    elif isinstance(Routers, utils_torch.PyObj):
        RoutersParsed = utils_torch.EmptyPyObj()
        for Name, Router in ListAttrsAndValues(Routers):
            setattr(RoutersParsed, Name, ParseRouterDynamic(Router, ObjRefList, **kw))
        return RoutersParsed
    else:
        raise Exception()

def ParseRouterStatic(Router, InPlace=True, **kw):
    if InPlace:
        for Index, Routing in enumerate(Router.Routings):
            Router.Routings[Index] = ParseRoutingStatic(Routing)
        _Router = Router
    else:
        RouterParsed = utils_torch.json.CopyPyObj(Router)
        RoutingsParsed = []
        for Index, Routing in enumerate(Router.Routings):
            RoutingsParsed.append(ParseRoutingStatic(Routing))
        setattr(RouterParsed, "Routings", RoutingsParsed)

        Router = RouterParsed
    EnsureAttrs(Router, "In", default=[])
    EnsureAttrs(Router, "Out", default=[])

    CheckRoutingsInputOutputNum(Router)
    utils_torch.parse.ParsePyObjStatic(Router, InPlace=True, **kw)
    return Router

def CheckRoutingsInputOutputNum(Router):
    for Index, Routing in enumerate(Router.Routings):
        if isinstance(Routing.Module, utils_torch.PyObj):
            RoutingInNum = len(Routing.In)
            ModuleInNum = len(Routing.Module.In)
            if RoutingInNum != ModuleInNum:
                raise Exception("Routing inputs %d param to its Module, which accepts %d param."%(RoutingInNum, ModuleInNum))
            RoutingOutNum = len(Routing.Out)
            ModuleOutNum = len(Routing.Module.Out)
            if RoutingOutNum != ModuleOutNum:
                raise Exception("Routing.Module ouputs %d param, whilc Routing accepts %d param."^(ModuleOutNum, RoutingOutNum))

def ParseRouterDynamic(Router, ObjRefList=[], InPlace=False, **kw):
    assert isinstance(Router, utils_torch.PyObj), "Object %s is not a Router."%Router
    RouterParsed = utils_torch.parse.ParsePyObjDynamic(
        Router, ObjRefList=ObjRefList, InPlace=InPlace, RaiseFailedParse=True, **kw
    )
    for routing in RouterParsed.Routings:
        #if hasattr(routing, "OnlineParseAttrs"):
        if "RepeatTime" not in routing.OnlineParseAttrs:
            routing.cache.RepeatTime = routing.RepeatTime
        if "Condition" not in routing.OnlineParseAttrs:
            routing.cache.Condition = routing.Condition
        if "InheritStates" not in routing.OnlineParseAttrs:
            routing.cache.InheritStates = routing.InheritStates
        # else:
        #     routing.cache.RepeatTime = routing.RepeatTime
        #     routing.cache.Condition = routing.Condition
        #     routing.cache.InheritStates = routing.InheritStates
        routing.cache.InDict = utils_torch.ToDict(routing.InDict)
    return RouterParsed

def ParseRoutingStatic(Routing):
    if not isinstance(Routing, str):
        return Routing

    _Routing = Routing
    param = utils_torch.EmptyPyObj()
    # notice that there might be . in _Routing string.
    SetAttrs(param, "Str", value=_Routing.replace("&", "(At)"))
    # if param.Str in ['DataBatch, Name=Input |--> (At)FilterFromDict |--> ModelInput']:
    #     print("aaa")
    Routing = re.sub(" ", "", Routing) # remove all spaces
    Routing = Routing.split("||")
    MainRouting = Routing[0] 
    if len(Routing) > 1:
        Attrs = Routing[1:]
    else:
        Attrs = []
    _MainRouting = MainRouting
    MainRouting = MainRouting.split("|-->")
    if len(MainRouting) != 3:
        if len(MainRouting)==2:
            if MainRouting[0].startswith("&") and not MainRouting[1].startswith("&"):
                MainRouting = ["", MainRouting[0], MainRouting[1]]
            elif not MainRouting[0].startswith("&") and MainRouting[1].startswith("&"):
                MainRouting = [MainRouting[0], MainRouting[1], ""]
            else:
                raise Exception("Cannot parse routing: %s"%_Routing)
        elif len(MainRouting)==1:
            MainRouting = ["", MainRouting[0], ""]
        else:
            raise Exception("Cannot parse routing: %s"%_Routing)
    In = MainRouting[0]
    Module = MainRouting[1]
    Out = MainRouting[2]

    if In=="":
       param.In = []
    else:
        param.In = In.rstrip(",").split(",")

    InList = []
    InDict = {}
    for Index, _Input in enumerate(param.In):
        _Input = _Input.split("=")
        if len(_Input)==2:
            Key = utils_torch.RemoveHeadTailWhiteChars(_Input[0])
            Value = utils_torch.RemoveHeadTailWhiteChars(_Input[1])
            try:
                ValueEval = eval(Value)
                # Bug to be fixed: Cases where value is synonymous with local variables here.
                Value = ValueEval
            except Exception:
                pass
            InDict[Key] = Value
        else:
            InList.append(_Input[0])
    param.In = InList
    param.InDict = InDict

    if Out=="":
        param.Out = []
    else:
        param.Out = []
        param.Out = Out.rstrip(",").split(",")
    param.Module = Module

    for Attr in Attrs:
        Attr = Attr.split("=")
        if len(Attr)==1:
            _Attr = Attr[0]
            if _Attr in ["InheritStates"]:
                Value = True
            else:
                raise Exception(_Attr)
        elif len(Attr)==2:
            _Attr, Value = Attr[0], Attr[1]
        else:
            raise Exception(len(Attr))

        if _Attr in ["repeat", "Repeat", "RepeatTime"]:
            _Attr = "RepeatTime"
        setattr(param, _Attr, Value)
    
    

    EnsureAttrs(param, "RepeatTime", value=1)
    param.cache.RepeatTime = param.RepeatTime

    EnsureAttrs(param, "Condition", value=True)
    if param.Condition is None:
        param.Condition = True

    EnsureAttrs(param, "InheritStates", value=False)
    if param.InheritStates is None:
        param.InheritStates = False

    SetOnlineParseAttrsForRouter(param)

    return param

def SetOnlineParseAttrsForRouter(routing):
    routing.OnlineParseAttrs = {}
    for Attr, Value in ListAttrsAndValues(routing):
        #if isinstance(Value, str) and "%" in Value: # Dynamic Parse
        if isinstance(Value, str) and Value.startswith("%"):
                routing.OnlineParseAttrs[Attr] = Value

    routing.InDictOnlineParseAttrs = {}
    for Key, Value in routing.InDict.items():
        #if isinstance(Value, str) and '%' in Value:
        if isinstance(Value, str) and Value.startswith("%"):
            #routing.InDictOnlineParseAttrs[Key] = Value[1:]
            routing.InDictOnlineParseAttrs[Key] = Value

    if len(routing.OnlineParseAttrs)>0 or len(routing.InDictOnlineParseAttrs)>0:
        routing.HasOnlineParseAttrs = True
    else:
        routing.HasOnlineParseAttrs = False
        #delattr(routing, "OnlineParseAttrs")
        delattr(routing, "InDictOnlineParseAttrs")

def ParseRoutingAttrsOnline(routing, States):
    #utils_torch.parse.RedirectPyObj(Routing, States)
    for attr, value in routing.OnlineParseAttrs.items():
        value = GetAttrs(routing, attr)    
        #value = re.sub("(%\w+)", lambda matchedStr:"getattr(States, %s)"%matchedStr[1:], value)
        value = eval(value.replace("%", "States."))
        setattr(routing.cache, attr, value)
    for attr, value in routing.InDictOnlineParseAttrs.items():
        routing.cache.InDict[attr] = eval(value.replace("%", "States."))
    return routing

class RouterStatic(utils_torch.PyObj):
    def FromPyObj(self, Obj):
        self.FromPyObj(utils_torch.router.ParseRouterStatic(Obj, InPlace=False))
    def ToRouterDynamic(self, **kw):
        return utils_torch.router.ParseRouterDynamic(self, **kw)
