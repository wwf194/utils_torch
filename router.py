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
    return RouterParsed

def ParseRoutingAttrsDynamic(Routing, States):
    #utils_torch.parse.RedirectPyObj(Routing, States)
    for attr in Routing.OnlineParseAttrs:
        value = GetAttrs(Routing, attr)    
        #value = re.sub("(%\w+)", lambda matchedStr:"getattr(States, %s)"%matchedStr[1:], value)
        value = eval(value.replace("%", "States."))     
        SetAttrs(Routing.cache, attr, value)
    return Routing

def ParseRoutingStatic(Routing):
    if not isinstance(Routing, str):
        return Routing
    _Routing = Routing
    param = utils_torch.EmptyPyObj()
    # notice that there might be . in _Routing string.
    SetAttrs(param, "Str", value=_Routing.replace("&", "(At)"))
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
    for Index, Input in enumerate(param.In):
        Input = Input.split("=")
        if len(Input)==2:
            Key = utils_torch.RemoveHeadTailWhiteChars(Input[0])
            Value = utils_torch.RemoveHeadTailWhiteChars(Input[1])
            try:
                ValueEval = eval(Value)
                Value = ValueEval
            except Exception:
                pass
            InDict[Key] = Value
        else:
            InList.append(Input[0])
    param.In = InList
    param.InNamed = InDict

    if Out=="":
        param.Out = []
    else:
        param.Out = []
        param.Out = Out.rstrip(",").split(",")
    param.Module = Module

    for Attr in Attrs:
        Attr = Attr.split("=")
        if len(Attr)!=2:
            raise Exception()
        _Attr, value = Attr[0], Attr[1]

        if _Attr in ["repeat"]:
            _Attr = "RepeatTime"
        setattr(param, _Attr, value)

    EnsureAttrs(param, "RepeatTime", value=1)

    param.cache.RepeatTime = param.RepeatTime
    param.OnlineParseAttrs = []
    for attr, value in ListAttrsAndValues(param):
        if isinstance(value, str):
            if value[0]=="%": # Dynamic Parse
                param.OnlineParseAttrs.append(attr)

    return param