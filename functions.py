import functools
import utils_torch
from utils_torch.attr import *

def StackFunctions(FunctionList, *Functions, Inverse=False, InputNum=1):
    if isinstance(FunctionList, list):
        if len(Functions)>0:
            raise Exception()
        Functions = FunctionList
    else:
        Functions = [FunctionList, *Functions]
    
    if len(Functions)==1:
        return Functions[0]

    if not Inverse:
        # Function at head of list is called earlier.
        # return functools.reduce(lambda f, g: lambda x: g(f(x)), Functions, lambda x: x)
        if InputNum == 0:
            _Functions = functools.reduce(lambda f, g: lambda x: g(f(x)), Functions[1:])
            return lambda :_Functions(Functions[0]())
        elif InputNum == 1:
            return functools.reduce(lambda f, g: lambda x: g(f(x)), Functions)
        else:
            raise Exception("To Be Implemented")
    else:
        # Function at tail of list is called earlier
        return functools.reduce(lambda f, g: lambda x: f(g(x)), Functions)
StackFunction = StackFunctions

def EmptyFunction(*Args, **KwArgs):
    return
NullFunction = EmptyFunction

def ParseFunctionParamsStatic(paramList):
    for index, param in enumerate(paramList):
        paramList[index] = ParseFunctionParamStatic(param)

def Call(Callable, *Args, **kw):
    if utils_torch.IsPyObj(Callable):
        Output = Callable(*Args, **kw)
        if isinstance(Output, list):
            return tuple(Output)
        return Output
    else:
        return Callable(*Args, **kw)

def ParseFunctionParamStatic(param, InPlace=False):
    if callable(param) and not utils_torch.IsPyObj(param):
        return [param, []]
    elif isinstance(param, str):
        return [param, []]
    elif utils_torch.IsListLike(param):
        if len(param)==0:
            param.append([])
        return param
    else:
        raise Exception(type(param))

def CallFunctions(param, **kw):
    ContextDict = kw
    Outputs = []
    if isinstance(param, utils_torch.PyObj):
        param = GetAttrs(param)
    if isinstance(param, utils_torch.PyObj): # Call one function
        Output = _CallFunction(param, ContextDict)
        Outputs.append(Output)
    elif isinstance(param, list) or utils_torch.IsListLikePyObj(param): # Call a cascade of functions
        for _param in param:
            Output = _CallFunction(_param, ContextDict)
            Outputs.append(Output)
    elif isinstance(param, str):
        Output = _CallFunction([param], ContextDict)
        Outputs.append(Output)
    elif callable(param) and not utils_torch.IsPyObj(): # Already parsed to methods.
        Output = _CallFunction([param], ContextDict)
        Outputs.append(Output)        
    else:
        raise Exception(param)
    return Outputs

def CallFunction(param, ContextInfo={}):
    return _CallFunction(param, ContextInfo)

def _CallFunction(param, ContextInfo={}):
    ContextInfo.setdefault("__PreviousFunctionOutput__", None)
    if isinstance(param, str):
        param = [param]
    elif callable(param) and not utils_torch.IsPyObj(param):
        param = [param]

    if len(param)==1:
        param.append([])
    
    FunctionName = param[0]
    FunctionArgs = utils_torch.ToList(param[1])
    # if FunctionName in ["&#utils_torch.ExternalMethods.AddObjRefForParseRouters"]:
    #     print("aaa")
    Function = utils_torch.parse.ResolveStr(
        FunctionName,
        ContextInfo
    )
    PositionalArgs, KeyWordArgs = utils_torch.parse.ParseFunctionArgs(FunctionArgs, ContextInfo)
    FunctionOutput = Function(*PositionalArgs, **KeyWordArgs)    
    ContextInfo["__PreviousFunctionOutput__"] = FunctionOutput
    if FunctionOutput is None:
        return []
    else:
        return FunctionOutput

def CallGraph(Router, *InList, **InDict):
#def CallGraph(Router, InList, InDict=None, **kw):
    # Register Router Input
    if "__States__" in InDict:
        States = InDict["__States__"]
        InDict.pop("__States__")
    else:
        States = utils_torch.EmptyPyObj()
    
    Index = 0
    for Key in Router.In:
        if Key in InDict:
            States[Key] = InDict[Key]
        else:
            States[Key] = InList[Index]
            Index += 1
    for routingIndex, routing in enumerate(Router.Routings):
        # if isinstance(Routing, list):
        #     CallFunction(Routing, **kw)
        if routing.HasOnlineParseAttrs:
            utils_torch.router.ParseRoutingAttrsOnline(routing, States)
        if routing.cache.Condition:
            for TimeIndex in range(routing.cache.RepeatTime):
                # Prepare Module InputList.
                InputList = []
                for Index, Key in enumerate(routing.In):
                    InputList.append(States[Key])
                
                # InputDict = Routing.InDict.ToDict()
                # for Key, Value in InputDict.items():
                #     if isinstance(Value, str) and Value.startswith("%"):
                #         InputDict[Key] = States[Value[1:]]
                InputDict = routing.cache.InDict
                #InputList = utils_torch.parse.FilterFromPyObj(States, Routing.In)
                # Routing.Module is a router
                if isinstance(routing.Module, utils_torch.PyObj):
                    if routing.cache.InheritStates:
                        OutputList = CallGraph(routing.Module, *InputList, __States__=States, **InputDict)
                    else:
                        OutputList = CallGraph(routing.Module, *InputList, **InputDict)
                    # if len(Routing.InDict) > 0:
                    #     #raise Exception(Routing.InDict)
                    #     OutputList = CallGraph(Routing.Module, *InputList, __States__=_States, **InputDict)
                    # else:
                    #     OutputList = CallGraph(Routing.Module, *InputList, __States__=_States)
                else: # Routing.Module is a method
                    OutputList = routing.Module(*InputList, **InputDict)
            RegisterModuleOutput(OutputList, routing, States)

    return utils_torch.parse.FilterFromPyObj(States, Router.Out)

def RegisterModuleOutput(OutputList, routing, States):
    # Process Module OutputList
    if len(routing.Out) > 1:
        for Index, Key in enumerate(routing.Out):
            if isinstance(OutputList, dict):
                OutputList = list(OutputList.values())
            States[Key] = OutputList[Index]
    elif len(routing.Out) == 1:
        if isinstance(OutputList, list):
            if len(OutputList)==1:
                States[routing.Out[0]] = OutputList[0]
            elif len(OutputList)>1:
                States[routing.Out[0]] = OutputList
            else:
                raise Exception()
        else:
            States[routing.Out[0]] = OutputList
    else:
        pass
    # utils_torch.parse.Register2PyObj(OutputList, States, Routing.Out)
