import re

import utils_torch
from utils_torch.attrs import *

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
    for Routing in RouterParsed.Routings:
        if "RepeatTime" not in Routing.OnlineParseAttrs:
            Routing.cache.RepeatTime = Routing.RepeatTime
        if "Condition" not in Routing.OnlineParseAttrs:
            Routing.cache.Condition = Routing.Condition
        if "InheritStates" not in Routing.OnlineParseAttrs:
            Routing.cache.InheritStates = Routing.InheritStates
        Routing.cache.InDict = utils_torch.PyObj(Routing.InDict)
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
    return param

def SetOnlineParseAttrsForRouter(routing):
    routing.OnlineParseAttrs = {}
    for attr, value in ListAttrsAndValues(routing):
        if isinstance(value, str):
            if value.startswith("%"): # Dynamic Parse
                routing.OnlineParseAttrs[attr] = value[1:]

    routing.InDictOnlineParseAttrs = {}
    for Key, Value in routing.InDict.items():
        if Value.startswith("%"):
            routing.InDictOnlineParseAttrs[Key] = Value[1:]

    if len(routing.OnlineParseAttrs)>0 or len(routing.InDictOnlineParseAttrs)>0:
        routing.HasOnlineParseAttrs = True
    else:
        routing.HasOnlineParseAttrs = False
        delattr(routing, "OnlineParseAttrs")
        delattr(routing, "InDictOnlineParseAttrs")

def ParseRoutingAttrsOnline(routing, States):
    #utils_torch.parse.RedirectPyObj(Routing, States)
    for attr, value in routing.OnlineParseAttrs.items():
        value = GetAttrs(routing, attr)    
        #value = re.sub("(%\w+)", lambda matchedStr:"getattr(States, %s)"%matchedStr[1:], value)
        value = eval(value.replace("%", "States."))
        setattr(routing.cache, attr, value)
    for attr, value in routing.InDictOnlineParseAttrs.items():
        routing.cache.InDict[attr] = States[value]
    return routing

import re

import utils_torch
from utils_torch.attrs import *

# class Router:
#     def __init__(self, param=None, data=None, **kw):
#         if param is not None:
#             self.param = param
#     def InitFromParam(self, param):
#         utils_torch.parse.ParseRoutingStr(param.Routings)
#     def forward(self, **kw):
#         param = self.param
#         States = utils_torch.EmptyPyObj()
#         for name in param.In:
#             setattr(States, name, kw[name])
#         for Routing in param.Routings:
#             utils_torch.parse.ParseRouterDynamic(Routing, States)
#             for TimeIndex in range(Routing.RepeatTime):
#                 InputDict = utils_torch.parse.FilterFromPyObj(States, Routing.In)
#                 OutObj = Routing.Module(InputDict)
#                 utils_torch.parse.Register2PyObj(OutObj, States, Routing.Out)
#         OutDict = {}
#         for name in param.Out:
#             OutDict[name] = getattr(States, name)
#         return OutDict

# def BuildRouter(param):
#     return utils_torch.Router.Router(param)

class RouterStatic(utils_torch.PyObj):
    def FromPyObj(self, Obj):
        self.FromPyObj(utils_torch.router.ParseRouterStatic(Obj, InPlace=False))
    def ToRouterDynamic(self, **kw):
        return utils_torch.router.ParseRouterDynamic(self, **kw)
