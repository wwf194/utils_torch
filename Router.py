import re

from utils_torch.attrs import *
from utils_torch.python import *

class Router:
    def __init__(self, param=None):
        if param is not None:
            self.param = param
    def InitFromParam(self, param):
        ParseRoutersStr(param.Routings)
    def forward(self, **kw):
        param = self.param
        States = utils_torch.EmptyPyObj()
        for name in param.In:
            setattr(States, name, kw[name])
        for Routing in param.Routings:
            ParseRouterDynamic(Routing, States)
            for time in range(Routing.RepeatTime):
                InputDict = FilterDictFromPyObj(Routing.In)
                OutObj = Routing.Module(InputDict)
                RegisterObj2PyObj(OutObj, States, Routing.Out)
        OutDict = {}
        for name in param.Out:
            OutDict[name] = getattr(States, name)
        return OutDict

def BuildRouter(param):
    return utils_torch.Router.Router(param)

build_signal_flow = BuildRouter

def ParseRouters(Param, RedirectTarget):
    if isinstance(RedirectTarget, list):
        utils_torch.parse.Redirect2PyObjList(Param, RedirectTarget)
    elif isinstance(RedirectTarget, utils_torch.json.PyObj):
        utils_torch.parse.Redirect2PyObj(Param, RedirectTarget)
    else:
        raise Exception()

def ParseRouterDynamic(Routing, States):
    #utils_torch.parse.RedirectPyObj(Routing, States)
    for attrs in Routing.DynamicParseAttrs:
        value = GetAttrs(Routing, attrs)    
    value = re.sub("([%]\w+)", lambda matchedStr:"getattr(States, %s)"%matchedStr[1:], value)
    SetAttrs(Routing, attrs, eval(value))

def ParseRoutersStr(Routings):
    if not isinstance(Routings, list):
        raise Exception()
    for Index, Routing in enumerate(Routings):
        Routings[Index] = _ParseRouterStr(Routing)

parse_computation_node_flows = ParseRoutersStr

def _ParseRouterStr(Routing):
    Routing = re.sub(" ", "", Routing) # remove all spaces
    param = utils_torch.EmptyPyObj()
    Routing = Routing.split("||")
    MainRouting = Routing[0] 
    if len(Routing) > 1:
        Params = Routing[1:]
    else:
        Params = []

    MainRouting = MainRouting.split("|-->")
    if len(MainRouting) != 3:
        raise Exception()
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