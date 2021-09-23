import re

import utils_torch
from utils_torch.attrs import *

def ParseRouterStaticAndDynamic(Router, **kw):
    ParseRouterStatic(Router, InPlace=True)
    return ParseRouterDynamic(Router, **kw)

ParseRouter = ParseRouterStaticAndDynamic

def ParseRoutersDynamic(Routers, ObjRefList=[], **kw):
    if isinstance(Routers, list):
        RouterParsed = []
        for Router in Routers:
            RouterParsed.append(ParseRouterDynamic(Router, ObjRefList, **kw))
        return RouterParsed
    elif isinstance(Routers, utils_torch.json.PyObj):
        RoutersParsed = utils_torch.EmptyPyObj()
        for Name, Router in ListAttrsAndValues(Routers):
            setattr(RoutersParsed, Name, ParseRouterDynamic(Router, ObjRefList, **kw))
        return RoutersParsed
    else:
        raise Exception()

def ParseRouterStatic(Router, InPlace=True, **kw):
    if InPlace:
        for Index, Routing in enumerate(Router.Routings):
            #RoutingsParsed.append(ParseRoutingStatic(Routing))
            Router.Routings[Index] = ParseRoutingStatic(Routing)
        CheckRoutingsInputOutputNum(Router)
        return Router
    else:
        RouterParsed = utils_torch.json.CopyPyObj(Router)
        RoutingsParsed = []
        for Index, Routing in enumerate(Router.Routings):
            RoutingsParsed.append(ParseRoutingStatic(Routing))
        setattr(RouterParsed, "Routings", RoutingsParsed)
        CheckRoutingsInputOutputNum(RouterParsed)
        return RouterParsed

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
    if not isinstance(Router, utils_torch.json.PyObj):
        utils_torch.AddWarning("Object %s is not a Router."%Router)
    RouterParsed = utils_torch.json.EmptyPyObj()
    RouterParsed = utils_torch.parse.ParsePyObjDynamic(Router, ObjRefList=ObjRefList, InPlace=InPlace, RaiseFailedParse=True, **kw)

    return RouterParsed

def ParseRoutingAttrsDynamic(Routing, States):
    #utils_torch.parse.RedirectPyObj(Routing, States)
    for attr in Routing.DynamicParseAttrs:
        value = GetAttrs(Routing, attr)    
        #value = re.sub("(%\w+)", lambda matchedStr:"getattr(States, %s)"%matchedStr[1:], value)
        value = eval(value.replace("%", "States."))     
        SetAttrs(Routing.cache, attr, value)
    return Routing

def ParseRoutingStatic(Routing):
    if not isinstance(Routing, str):
        return Routing
    _Routing = Routing
    param = utils_torch.json.EmptyPyObj()
    param.cache = utils_torch.json.EmptyPyObj()
    SetAttrs(param, "Str", _Routing.replace("&", "(At)"))
    Routing = re.sub(" ", "", Routing) # remove all spaces
    Routing = Routing.split("||")
    MainRouting = Routing[0] 
    if len(Routing) > 1:
        Params = Routing[1:]
    else:
        Params = []
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
        else:
            raise Exception("Cannot parse routing: %s"%_Routing)
    In = MainRouting[0]
    Module = MainRouting[1]
    Out = MainRouting[2]

    if In=="":
       param.In = []
    else:
        param.In = In.rstrip(",").split(",")
    if Out=="":
        param.Out = []
    else:
        param.Out = []
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

    param.cache.RepeatTime = param.RepeatTime

    param.DynamicParseAttrs = []
    for attr, value in ListAttrsAndValues(param):
        if isinstance(value, str):
            #if "%" in value:
            if value[0]=="%": # Dynamic Parse
                param.DynamicParseAttrs.append(attr)
    return param