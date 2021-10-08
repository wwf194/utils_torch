import functools
import utils_torch
from utils_torch.attrs import *
def ParseFunctionParamsStatic(paramList):
    for Index, param in enumerate(paramList):
        paramList[Index] = ParseFunctionParamStatic(param)

def ParseFunctionParamStatic(param, InPlace=False):
    if callable(param):
        return [param, []]
    elif isinstance(param, str):
        return [param, []]
    elif utils_torch.IsListLike(param):
        if len(param)==0:
            param.append([])
    else:
        raise Exception()

def StackFunction(FunctionList, *Functions, Inverse=False, InputNum=1):
    if isinstance(FunctionList, list):
        if len(Functions)>0:
            raise Exception()
        Functions = FunctionList
    else:
        Functions = [FunctionList, *Functions]
    
    if len(Functions)==1:
        return Functions[0]

    if not Inverse:
        # Function at head is called earlier.
        #return functools.reduce(lambda f, g: lambda x: g(f(x)), Functions, lambda x: x)
        if InputNum == 0:
            _Functions = functools.reduce(lambda f, g: lambda x: g(f(x)), Functions[1:])
            return lambda :_Functions(Functions[0]())
        elif InputNum == 1:
            return functools.reduce(lambda f, g: lambda x: g(f(x)), Functions)
        else:
            raise Exception("To Be Implemented")
    else:
        # Function at tail is called earlier
        return functools.reduce(lambda f, g: lambda x: f(g(x)), Functions)

def CallFunctions(param, **kw):
    ContextInfo = kw
    Outputs = []
    if isinstance(param, utils_torch.PyObj):
        param = GetAttrs(param)
    if isinstance(param, utils_torch.PyObj): # Call one function
        Output = _CallFunction(param, ContextInfo)
        Outputs.append(Output)
    elif isinstance(param, list) or utils_torch.IsListLikePyObj(param): # Call a cascade of functions
        for _param in param:
            Output = _CallFunction(_param, ContextInfo)
            Outputs.append(Output)
    elif isinstance(param, str):
        Output = _CallFunction([param], ContextInfo)
        Outputs.append(Output)
    else:
        raise Exception()
    return Outputs

def CallFunction(param, ContextInfo={}):
    return _CallFunction(param, ContextInfo)

def _CallFunction(param, ContextInfo={}):
    ContextInfo.setdefault("__PreviousFunctionOutput__", None)
    if isinstance(param, str):
        param = [param]
    if len(param)==1:
        param.append([])
    
    FunctionName = param[0]
    FunctionArgs = utils_torch.ToList(param[1])
    Function = utils_torch.parse.ResolveStrDict(
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

def CallGraph(Router, In, **kw):
    States = utils_torch.EmptyPyObj()
    # Register Router Input
    for Index, Key in enumerate(Router.In):
        States[Key] = In[Index]
    # utils_torch.parse.Register2PyObj(In, States, Router.In)

    # Run Router Routings
    for RoutingIndex, Routing in enumerate(Router.Routings):
        if isinstance(Routing, list):
            CallFunction(Routing, **kw)
        elif isinstance(Routing, utils_torch.PyObj):
            Routing = utils_torch.router.ParseRoutingAttrsDynamic(Routing, States)
            for TimeIndex in range(Routing.cache.RepeatTime):
                # Prepare Module InputList.
                InputList = []
                for Index, State in enumerate(Routing.In):
                    InputList.append(States[State])
                InputDict = Routing.InNamed
                for key, value in InputDict.items():
                    if value.startswith("%"):
                        InputDict[key] = States[value[1:]]
                #InputList = utils_torch.parse.FilterFromPyObj(States, Routing.In)
                if isinstance(Routing.Module, utils_torch.PyObj):
                    if len(Routing.InNamed) > 0:
                        raise Exception(Routing.InNamed)
                    OutputList = CallGraph(Routing.Module, InputList)
                else:
                    OutputList = Routing.Module(*InputList, **InputDict)
                
                # Process Module OutputList
                if len(Routing.Out) > 1:
                    for Index, Key in enumerate(Routing.Out):
                        States[Key] = OutputList[Index]
                elif len(Routing.Out) == 1:
                    if isinstance(OutputList, list):
                        if len(OutputList)==1:
                            States[Routing.Out[0]] = OutputList[0]
                        elif len(OutputList)>1:
                            States[Routing.Out[0]] = OutputList
                        else:
                            raise Exception()
                    else:
                        States[Routing.Out[0]] = OutputList
                else:
                    pass
                # utils_torch.parse.Register2PyObj(OutputList, States, Routing.Out)
        else:
            raise Exception()
    return utils_torch.parse.FilterFromPyObj(States, Router.Out)