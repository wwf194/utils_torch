import os
import re
import sys
import functools
import threading
import time
import warnings
import pickle
import random
import importlib
from typing import Iterable, List
#import pynvml
#from pynvml.nvml import nvmlDeviceOnSameBoard
from types import SimpleNamespace

#import timeout_decorator
import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
from matplotlib import pyplot as plt

from inspect import getframeinfo, stack

from utils_torch.attrs import *
from utils_torch.files import *

import traceback

def Main(CommandArgs):
    TaskFilePath = CommandArgs.TaskFile # All sciprt loads a task file, and keep doing tasks in it.
    if CommandArgs.task in ["CleanLog", "CleanLog", "cleanlog"]:
        CleanLog()
    elif CommandArgs.task in ["CleanFigure"]:
        CleanFigures()
    elif CommandArgs.task in ["DoTasksFromFile"]:
        TaskObj = utils_torch.LoadTaskFile(TaskFilePath)
        Tasks = getattr(TaskObj, CommandArgs.TaskName)
        if not CommandArgs.IsDebug:
            try: # catch all unhandled exceptions
                utils_torch.DoTasks(Tasks, ObjRoot=utils_torch.GetGlobalParam())
            except Exception:
                utils_torch.AddError(traceback.format_exc())
                raise Exception()
        else:
            utils_torch.DoTasks(Tasks, ObjRoot=utils_torch.GetGlobalParam())
    elif CommandArgs.task in ["TotalLines"]:
        utils_torch.CalculateGitProjectTotalLines()
    elif CommandArgs.task in ["QuickScript"]:
        CommandArgs.QuickScript(CommandArgs)
    else:
        raise Exception("Inavlid Task: %s"%CommandArgs.task)

def CleanLog():
    utils_torch.files.RemoveAllFilesAndDirs("./log/")

def CleanFigures():
    utils_torch.files.RemoveMatchedFiles("./", r".*\.png")

def ParseTaskList(TaskList, InPlace=True, **kw):
    TaskListParsed = []
    for Index, Task in enumerate(TaskList):
        if isinstance(Task, str):
            TaskParsed = utils_torch.PyObj({
                "Type": Task,
                "Args": {}
            })
            if InPlace:
                TaskList[Index] = TaskParsed
            else:
                TaskListParsed.append(TaskParsed)
        elif utils_torch.IsDictLikePyObj(Task):
            if hasattr(Task, "Type") and hasattr(Task, "Args"):
                if InPlace:
                    pass
                else:
                    TaskListParsed.append(Task)
            else:
                for key, value in ListAttrsAndValues(Task):
                    TaskParsed = utils_torch.PyObj({
                        "Type": key, "Args": value
                    })

                    if InPlace:
                        TaskList[Index] = TaskParsed
                    else:
                        TaskListParsed.append(TaskParsed)
        elif utils_torch.IsListLikePyObj(Task) or isinstance(Task, list) or isinstance(Task, tuple):
            TaskParsed = utils_torch.PyObj({
                "Type": Task[0],
                "Args": Task[1]
            })
            if InPlace:
                TaskList[Index] = TaskParsed
            else:
                TaskListParsed.append(TaskParsed)
        else:
            raise Exception(type(Task))
        
    if InPlace:
        return TaskList
    else:
        return utils_torch.PyObj(TaskListParsed)

def ParseTaskObj(TaskObj, Save=True, **kw):
    kw.setdefault("ObjRoot", utils_torch.GetGlobalParam())
    if isinstance(TaskObj, str):
        TaskObj = utils_torch.parse.ResolveStr(TaskObj, **kw)
    if utils_torch.IsDictLikePyObj(TaskObj):
        if hasattr(TaskObj, "__Tasks__"):
            TaskObj.__Tasks__ = ParseTaskList(TaskObj.__Tasks__, **kw)
            TaskList = TaskObj.__Tasks__
        else:
            TaskObj.__Tasks__ = ParseTaskList(ListAttrsAndValues(TaskObj), InPlace=False, **kw)
            for Attr, Value in ListAttrsAndValues(TaskObj, Exceptions=["__Tasks__"]):
                delattr(TaskObj, Attr)
            TaskList = TaskObj.__Tasks__
    elif utils_torch.IsListLikePyObj(TaskObj):
        TaskObj.__Tasks__ = ParseTaskList(TaskObj, InPlace=False, **kw)
        delattr(TaskObj, "__value__")
        TaskList = TaskObj.__Tasks__
    else:
        raise Exception()
    TaskObj.SetResolveBase(True)
    TaskList.SetResolveBase(False)
    for Index, Task in enumerate(TaskList):
        Task.SetResolveBase() # So that "&" in each Task resolves to the task object it is inside.

    if Save:
        utils_torch.json.PyObj2JsonFile(TaskList, utils_torch.GetMainSaveDir() + "task_loaded.jsonc")
    utils_torch.parse.ParsePyObjStatic(TaskObj, ObjCurrent=TaskList, ObjRoot=utils_torch.GetGlobalParam(), InPlace=True)
    if Save:
        utils_torch.json.PyObj2JsonFile(TaskList, utils_torch.GetMainSaveDir() + "task_parsed.jsonc")
    return TaskObj


def DoTasks(Tasks, **kw):
    if not kw.get("DoNotChangeObjCurrent"):
        kw["ObjCurrent"] = Tasks
    if isinstance(Tasks, str) and "&" in Tasks:
        Tasks = utils_torch.parse.ResolveStr(Tasks, **kw)
    Tasks = utils_torch.ParseTaskObj(Tasks)

    In = kw.get("In")
    if In is not None:
        Tasks.cache.In = utils_torch.PyObj(In)
    for Index, Task in enumerate(Tasks.__Tasks__):
        if not kw.get("DoNotChangeObjCurrent"):
            kw["ObjCurrent"] = Task
        #utils_torch.EnsureAttrs(Task, "Args", default={})
        utils_torch.DoTask(Task, **kw)

def DoTask(Task, **kw):
    ObjRoot = kw.setdefault("ObjRoot", None)
    ObjCurrent = kw.setdefault("ObjCurrent", None)
    #Task = utils_torch.parse.ParsePyObjDynamic(Task, RaiseFailedParse=False, InPlace=False, **kw)
    TaskType = Task.Type
    TaskArgs = Task.Args
    if isinstance(TaskArgs, str) and "&" in TaskArgs:
        TaskArgs = utils_torch.parse.ResolveStr(TaskArgs, **kw)
    if TaskType in ["BuildObjFromParam", "BuildObjectFromParam"]:
        BuildObjFromParam(TaskArgs, **kw)
    elif TaskType in ["FunctionCall"]:
        utils_torch.CallFunctions(TaskArgs, **kw)
    elif TaskType in ["CallGraph"]:
        if hasattr(TaskArgs, "Router"):
            Router = TaskArgs.Router
        else:
            Router = TaskArgs
        if isinstance(Router, str):
            Router = utils_torch.parse.ResolveStr(Router)
        # Require that router is already parsed.
        #RouterParsed = utils_torch.router.ParseRouterStaticAndDynamic(Router, ObjRefList=[Router], **kw)
        InParsed = utils_torch.parse.ParsePyObjDynamic(TaskArgs.In, RaiseFailedParse=True, InPlace=False, **kw)
        #InParsed = utils_torch.parse.ParsePyObjDynamic(Router, RaiseFailedParse=True, InPlace=False, **kw)
        utils_torch.CallGraph(Router, InParsed)
    elif TaskType in ["RemoveObj"]:
        RemoveObj(TaskArgs, **kw)
    elif TaskType in ["LoadObjFromFile"]:
        LoadObjFromFile(TaskArgs, **kw)
    elif TaskType in ["LoadObj"]:
        utils_torch.LoadObj(TaskArgs, **kw)
    elif TaskType in ["AddLibraryPath"]:
        AddLibraryPath(TaskArgs)
    elif TaskType in ["LoadJsonFile"]:
        LoadJsonFile(TaskArgs)
    elif TaskType in ["LoadParamFile"]:
        utils_torch.LoadParamFromFile(TaskArgs, ObjRoot=utils_torch.GetGlobalParam())
    elif TaskType in ["ParseParam", "ParseParamStatic"]:
        utils_torch.parse.ParseParamStatic(TaskArgs)
    elif TaskType in ["ParseParamDynamic"]:
        utils_torch.parse.ParseParamDynamic(TaskArgs)
    elif TaskType in ["BuildObj"]:
        utils_torch.BuildObj(TaskArgs, **kw)
    elif TaskType in ["BuildObjFromFile", "BuildObjectFromFile"]:
        utils_torch.BuildObjFromFile(TaskArgs, ObjRoot=utils_torch.GetGlobalParam())
    elif TaskType in ["BuildObjFromParam", "BuildObjectFromParam"]:
        utils_torch.BuildObjFromParam(TaskArgs, ObjRoot=utils_torch.GetGlobalParam())
    elif TaskType in ["SetTensorLocation"]:
        SetTensorLocation(TaskArgs)
    elif TaskType in ["Train"]:
        utils_torch.train.Train(
            TaskArgs,
            ObjRoot=utils_torch.GetGlobalParam(),
            Logger=utils_torch.GetDataLogger()
        )
    elif TaskType in ["DoTasks"]:
        _TaskList = utils_torch.ParseTaskObj(TaskArgs, ObjRoot=utils_torch.GetGlobalParam())
        DoTasks(_TaskList, **kw)
    elif TaskType in ["SaveObj"]:
        utils_torch.SaveObj(TaskArgs, ObjRoot=utils_torch.GetGlobalParam())

    else:
        utils_torch.AddWarning("Unknown Task.Type: %s"%TaskType)
        raise Exception(TaskType)

def SetTensorLocation(Args):
    EnsureAttrs(Args, "Method", default="Auto")
    GlobalParam = utils_torch.GetGlobalParam()
    if HasAttrs(GlobalParam, "system.TensorLocation"):
        Location = GlobalParam.system.TensorLocation
    else:
        if Args.Method in ["Auto", "auto"]:
            Location = utils_torch.GetGPUWithLargestUseableMemory()
        else:
            raise Exception()

    for Obj in utils_torch.ListValues(utils_torch.GetGlobalParam().object):
        if hasattr(Obj, "SetTensorLocation"):
            Obj.SetTensorLocation(Location)
    SetAttrs(utils_torch.GetGlobalParam(), "system.TensorLocation", Location)

def BuildObjFromParam(Args, **kw):
    if isinstance(Args, utils_torch.PyObj):
        Args = GetAttrs(Args)

    if isinstance(Args, list):
        for Arg in Args:
            _BuildObjFromParam(Arg, **kw)
    elif isinstance(Args, utils_torch.PyObj):
        _BuildObjFromParam(Args, **kw)
    else:
        raise Exception()

def _BuildObjFromParam(Args, **kw):
    ParamPathList = utils_torch.ToList(Args.ParamPath)
    ModulePathList = utils_torch.ToList(Args.ModulePath)
    MountPathList = utils_torch.ToList(Args.MountPath)

    for ModulePath, ParamPath, MountPath, in zip(ModulePathList, ParamPathList, MountPathList):        
        param = utils_torch.parse.ResolveStr(ParamPath, **kw)
        #Class = eval(ModulePath)
        #Obj = Class(param)
        Class = utils_torch.parse.ParseClass(ModulePath)
        Obj = Class(param)
        # Module = utils_torch.ImportModule(ModulePath)
        # Obj = Module.__MainClass__(param)

        ObjRoot = kw.get("ObjRoot")
        ObjCurrent = kw.get("ObjCurrent")
        
        MountPath = MountPath.replace("/&", "&")
        MountPath = MountPath.replace("&^", "ObjRoot.")
        MountPath = MountPath.replace("&*", "ObjCurrent.cache.__object__.")
        MountPath = MountPath.replace("&", "ObjCurrent.")

        MountPathList = MountPath.split(".")
        SetAttrs(eval(MountPathList[0]), MountPathList[1:], Obj)

def BuildObjFromFile(Args, **kw):
    if isinstance(Args, utils_torch.PyObj):
        Args = GetAttrs(Args)
    if isinstance(Args, list):
        for Arg in Args:
            _BuildObjFromFile(Arg, **kw)
    elif isinstance(Args, utils_torch.PyObj):
        _BuildObjFromFile(Args, **kw)
    else:
        raise Exception()

def _BuildObjFromFile(Args, **kw):
    ParamFilePathList = utils_torch.ToList(Args.ParamFilePath)
    ModulePathList = utils_torch.ToList(Args.ModulePath)
    MountPathList = utils_torch.ToList(Args.MountPath)

    for ModulePath, ParamFilePath, MountPath, in zip(ModulePathList, ParamFilePathList, MountPathList):        
        param = utils_torch.json.JsonFile2PyObj(ParamFilePath)
        Class = utils_torch.parse.ParseClass(ModulePath)
        Obj = Class(param)

        ObjRoot = kw.get("ObjRoot")
        ObjCurrent = kw.get("ObjCurrent")
        
        MountPath = MountPath.replace("/&", "&")
        MountPath = MountPath.replace("&^", "ObjRoot.")
        MountPath = MountPath.replace("&*", "ObjCurrent.cache.__object__.")
        MountPath = MountPath.replace("&", "ObjCurrent.")

        MountPathList = MountPath.split(".")
        SetAttrs(eval(MountPathList[0]), MountPathList[1:], Obj)

def BuildObj(Args, **kw):
    if isinstance(Args, utils_torch.PyObj):
        Args = GetAttrs(Args)

    if isinstance(Args, list) or utils_torch.IsListLikePyObj(Args):
        for Arg in Args:
            _BuildObj(Arg, **kw)
    elif isinstance(Args, utils_torch.PyObj):
        _BuildObj(Args, **kw)
    else:
        raise Exception()

def _BuildObj(Args, **kw):
    ModulePathList = utils_torch.ToList(Args.ModulePath)
    MountPathList = utils_torch.ToList(Args.MountPath)

    for ModulePath, MountPath in zip(ModulePathList, MountPathList):
        # Module = utils_torch.ImportModule(_ModulePath)
        #Obj = Module.__MainClass__()
        # Class = eval(ModulePath)
        Class = utils_torch.parse.ParseClass(ModulePath)
        Obj = Class()

        ObjRoot = kw.get("ObjRoot")
        ObjCurrent = kw.get("ObjCurrent")
        
        MountPath = MountPath.replace("&^", "ObjRoot.")
        MountPath = MountPath.replace("&*", "ObjCurrent.cache.__object__.")
        MountPath = MountPath.replace("&", "ObjCurrent.")

        MountPathList = MountPath.split(".")
        SetAttrs(eval(MountPathList[0]), MountPathList[1:], Obj)

def RemoveObj(Args, **kw):
    if isinstance(Args, utils_torch.PyObj):
        Args = GetAttrs(Args)

    if isinstance(Args, list):
        for Arg in Args:
            _RemoveObj(Arg, **kw)
    elif isinstance(Args, utils_torch.PyObj):
        _RemoveObj(Args, **kw)
    else:
        raise Exception()

def _RemoveObj(Args, **kw):
    MountPathList = utils_torch.ToList(Args.MountPath)

    for MountPath in MountPathList:
        ObjRoot = kw.get("ObjRoot")
        ObjCurrent = kw.get("ObjCurrent")
        
        MountPath = MountPath.replace("/&", "&")
        MountPath = MountPath.replace("&^", "ObjRoot.")
        MountPath = MountPath.replace("&*", "ObjCurrent.cache.__object__.")
        MountPath = MountPath.replace("&", "ObjCurrent.")
        MountPathList = MountPath.split(".")
        RemoveAttrs(eval(MountPathList[0]), MountPathList[1:])

def SaveObj(Args, **kw):
    SaveObjList = utils_torch.ToList(Args.SaveObj)
    SaveDirList = utils_torch.ToList(Args.SaveDir)

    for SaveObj, SaveDir in zip(SaveObjList, SaveDirList):
        if SaveDir in ["auto", "Auto"]:
            SaveDir = utils_torch.GetMainSaveDirForModel()
        Obj = utils_torch.parse.ResolveStr(SaveObj, **kw)
        Obj.Save(SaveDir)

def LoadObj(Args, **kw):
    SourcePathList = utils_torch.ToList(Args.SourcePath)
    MountPathList = utils_torch.ToList(Args.MountPath)

    for SourcePath, MountPath in zip(SourcePathList, MountPathList):
        Obj = utils_torch.parse.ResolveStr(SourcePath, **kw)
        MountObj(MountPath, Obj, **kw)

def LoadObjFromFile(Args, **kw):
    SaveNameList = utils_torch.ToList(Args.SaveName)
    MountPathList = utils_torch.ToList(Args.MountPath)
    SaveDirList = utils_torch.ToList(Args.SaveDir)

    SaveDirParsedList = []
    for SaveDir in SaveDirList:
        SaveDirParsedList.append(utils_torch.parse.ResolveStr(SaveDir, **kw))


    for SaveName, SaveDir, MountPath in zip(SaveNameList, SaveDirParsedList, MountPathList):
        ParamPath = SaveDir + SaveName + ".param.jsonc"
        assert utils_torch.FileExists(ParamPath)
        param = utils_torch.json.JsonFile2PyObj(ParamPath)
        DataPath = SaveDir + SaveName + ".data"
        assert utils_torch.FileExists(DataPath)
        data = utils_torch.json.DataFile2PyObj(DataPath)
        Class = utils_torch.parse.ParseClass(param.ClassPath)
        Obj = Class(param, data, LoadDir=SaveDir)
        MountObj(MountPath, Obj, **kw)

def LoadTaskFile(FilePath="./task.jsonc"):
    TaskObj = utils_torch.json.JsonFile2PyObj(FilePath)
    return TaskObj

def LoadJsonFile(Args):
    if isinstance(Args, utils_torch.PyObj):
        Args = GetAttrs(Args)
    if isinstance(Args, dict):
        _LoadJsonFile(utils_torch.json.JsonObj2PyObj(Args))
    elif isinstance(Args, list):
        for Arg in Args:
            _LoadJsonFile(Arg)
    elif isinstance(Args, utils_torch.PyObj):
        _LoadJsonFile(Args)
    else:
        raise Exception()

def _LoadJsonFile(Args, **kw):
    Obj = utils_torch.json.JsonFile2PyObj(Args.FilePath)
    MountObj(Args.MountPath, Obj, **kw)

def AddLibraryPath(Args):
    if isinstance(Args, dict):
        _AddLibraryPath(Args)
    elif isinstance(Args, list):
        for Args_dict in Args:
            _AddLibraryPath(Args_dict)
    else:
        raise Exception()

def _AddLibraryPath(Args):
    # requires Args to be a dict.
    lib_name = Args['name']
    lib_path = Args['path']
    if lib_path=="!Getfrom_config":
        success = False
        for config_name, config_dict in utils_torch.GetGlobalParam().ConfigDicts.__dict__.items():
            if config_dict.get("libs") is not None:
                libs = config_dict["libs"]
                if libs.get(lib_name) is not None:
                    lib_path = libs[lib_name]["path"]
                    success = True
                    break
        if not success:
            utils_torch.AddWarning('add_lib failed: cannot find path to lib %s'%lib_name)
            return
    if os.path.exists(lib_path):
        if os.path.isdir(lib_path):
            sys.path.append(lib_path)
            utils_torch.AddLog("Added library <%s> from path %s"%(lib_name, lib_path))
        else:
            utils_torch.AddWarning('add_lib failed: path %s exists but is not a directory.'%lib_path)
    else:
        utils_torch.AddWarning('add_lib: invalid lib_path: ', lib_path)

def SaveObj(Args):
    Obj = utils_torch.parse.ResolveStr(Args.MountPath, ObjRoot=utils_torch.GetGlobalParam()),
    Obj.Save(SaveDir=Args.SaveDir)

def IsClassInstance(Obj):
    # It seems that in Python, all variables are instances of some class.
    return

import types
def IsFunction(Obj):
    return isinstance(Obj, types.FunctionType) \
        or isinstance(Obj, types.BuiltinFunctionType)

from collections.abc import Iterable   # import directly from collections for Python < 3.3
def IsIterable(Obj):
    if isinstance(Obj, Iterable):
        return True
    else:
        return False
def IsListLike(List):
    if isinstance(List, list):
        return True
    elif isinstance(List, utils_torch.PyObj) and List.IsListLike():
        return True
    else:
        return False

def RemoveStartEndEmptySpaceChars(Str):
    Str = re.match(r"\s*([\S].*)", Str).group(1)
    Str = re.match(r"(.*[\S])\s*", Str).group(1)
    return Str

RemoveHeadTailWhiteChars = RemoveStartEndEmptySpaceChars

def RemoveWhiteChars(Str):
    Str = re.sub(r"\s+", "", Str)
    return Str

def TensorType(data):
    return data.dtype

def NpArrayType(data):
    if not isinstance(data, np.ndarray):
        return "Not an np.ndarray, but %s"%type(data)
    return data.dtype

def List2NpArray(data):
    return np.array(data)

def Dict2GivenType(Dict, Type):
    if Type in ["PyObj"]:
        return utils_torch.PyObj(Dict)
    elif Type in ["Dict"]:
        return Dict
    else:
        raise Exception(Type)

def ToNpArray(data, DataType=np.float32):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data, dtype=DataType)
    elif isinstance(data, torch.Tensor):
        return Tensor2NpArray(data)
    else:
        raise Exception()


def ToTorchTensor(data):
    if isinstance(data, np.ndarray):
        return NpArray2Tensor(data)
    elif isinstance(data, list):
        return NpArray2Tensor(List2NpArray(data))
    else:
        raise Exception(type(data))

def Line2Square(data):
    DimensionNum = utils_torch.GetDimensionNum(data)
    if not DimensionNum == 1:
        raise Exception(DimensionNum)

    dataNum = data.shape[0]
    RowNum, ColNum = utils_torch.plot.ParseRowColNum(dataNum)
    mask = np.ones((RowNum, ColNum), dtype=np.bool8)

    maskNum = RowNum * ColNum - dataNum
    RowIndex, ColIndex = RowNum - 1, ColNum - 1 # Start from point at right bottom.
    
    for Index in range(maskNum):
        mask[RowIndex, ColIndex] = False
        ColIndex -= 1
    if maskNum > 0:
        data = np.concatenate([data, np.zeros(maskNum,dtype=data.dtype)])
    data = data.reshape((RowNum, ColNum))
    return data, mask

def FlattenNpArray(data):
    return data.flatten()

def EnsureFlatNp(data):
    return data.flatten()

EnsureFlat = EnsureFlatNp

def NpArray2Tensor(data, Location="cpu", DataType=torch.float32, RequiresGrad=False):
    data = torch.from_numpy(data)
    data = Tensor2GivenDataType(data, DataType)
    data = data.to(Location)
    data.requires_grad = RequiresGrad
    return data

def NpArray2List(data):
    return data.tolist()

def NpArray2Str(data):
    return np.array2string(data)

def ToStandardizeTorchDataType(DataType):
    if DataType in ["Float", "float"]:
        return torch.float32
    elif DataType in ["Double", "double"]:
        return torch.float64

def ToGivenDataTypeTorch(data, DataType=torch.float32):
    if data.dtype==DataType:
        return data
    else:
        return data.to(DataType)
Tensor2GivenDataType = ToGivenDataTypeTorch

def ParseDataTypeNp(DataType):
    if isinstance(DataType, str):
        # if DataType in ["np.float32"]:
        #     return np.float32
        # elif DataType in ["np.int8"]:
        #     return np.int8
        # else:
        #     raise Exception(DataType)
        #     # To Be Implemented
        return eval(DataType)
    else:
        return DataType

def ToGivenDataTypeNp(data, DataType):
    DataType = utils_torch.ParseDataTypeNp(DataType)
    return data.astype(DataType)

def TorchTensor2NpArray(data):
    data = data.detach().cpu().numpy()
    return data # data.grad will be lost.
Tensor2NpArray = TorchTensor2NpArray

def Tensor2Str(data):
    return NpArray2Str(Tensor2NpArray(data))

def Tensor2File(data, SavePath):
    EnsureFileDir(SavePath)
    np.savetxt(SavePath, utils_torch.Tensor2NpArray(data))

def Tensor2NumpyOrFloat(data):
    try:
        _data = data.item()
        return _data
    except Exception:
        pass
    data = data.detach().cpu().numpy()
    return data

def List2NpArray(data, Type=None):
    if Type is not None:
        return np.array(data, dtype=Type)
    else:
        return np.array(data)

def ToList(Obj):
    if isinstance(Obj, list):
        return Obj
    elif isinstance(Obj, np.ndarray):
        return Obj.tolist()
    elif isinstance(Obj, torch.Tensor):
        return NpArray2List(Tensor2NpArray(Obj))
    elif utils_torch.IsListLikePyObj(Obj):
        return Obj.ToList()
    elif isinstance(Obj, dict) or utils_torch.IsDictLikePyObj(Obj):
        raise Exception()
    else:
        return [Obj]

# def GetFunction(FunctionName, ObjRoot=None, ObjCurrent=None, **kw):
#     return eval(FunctionName.replace("&^", "ObjRoot.").replace("&", "ObjCurrent"))

def ContainAtLeastOne(List, Items, *args):
    if isinstance(Items, list):
        Items = [*Items, *args]
    else:
        Items = [Items, *args] 
    for Item in Items:
        if Item in List:
            return True
    return False

def ContainAll(List, Items, *args):
    if isinstance(Items, list):
        Items = [*Items, *args]
    else:
        Items = [Items, *args]   
    for Item in Items:
        if Item not in List:
            return False
    return True

import timeout_decorator

def CallFunctionWithTimeLimit(TimeLimit, Function, *Args, **ArgsKw):
    # TimeLimit: in seconds.
    event = threading.Event()

    FunctionThread = threading.Thread(target=NotifyWhenFunctionReturn, args=(event, Function, *Args), kwargs=ArgsKw)
    FunctionThread.setDaemon(True)
    FunctionThread.start()

    TimerThread = threading.Thread(target=NotifyWhenFunctionReturn, args=(event, ReturnInGivenTime, TimeLimit))
    TimerThread.setDaemon(True)
    # So that this thread will be forced to terminate with the thread calling this function.
    # Which does not satisfy requirement. We need this thread to terminate when this function returns.
    TimerThread.start()
    event.wait()
    return 

def NotifyWhenFunctionReturn(event, Function, *Args, **ArgsKw):
    Function(*Args, **ArgsKw)
    event.set()

def ReturnInGivenTime(TimeLimit, Verbose=True):
    # TimeLimit: float or int. In Seconds.
    if Verbose:
        utils_torch.AddLog("Start counding down. TimeLimit=%d."%TimeLimit)
    time.sleep(TimeLimit)
    if Verbose:
        utils_torch.AddLog("TimeLimit reached. TimeLimit=%d."%TimeLimit)
    return

def GetGPUWithLargestUseableMemory(TimeLimit=10, Default='cuda:0'):
    # GPU = [Default]
    # CallFunctionWithTimeLimit(TimeLimit, __GetGPUWithLargestUseableMemory, GPU)
    # return GPU[0]
    return _GetGPUWithLargestUseableMemory()

def __GetGPUWithLargestUseableMemory(List):
    GPU= _GetGPUWithLargestUseableMemory()
    List[0] = GPU
    utils_torch.AddLog("Selected GPU: %s"%List[0])

def _GetGPUWithLargestUseableMemory(Verbose=True): # return torch.device with largest available gpu memory.
    try:
        import pynvml
        pynvml.nvmlInit()
        GPUNum = pynvml.nvmlDeviceGetCount()
        GPUUseableMemory = []
        for GPUIndex in range(GPUNum):
            Handle = pynvml.nvmlDeviceGetHandleByIndex(GPUIndex) # sometimes stuck here.
            MemoryInfo = pynvml.nvmlDeviceGetMemoryInfo(Handle)
            GPUUseableMemory.append(MemoryInfo.free)
        GPUUseableMemory = np.array(GPUUseableMemory, dtype=np.int64)
        GPUWithLargestUseableMemoryIndex = np.argmax(GPUUseableMemory)    
        if Verbose:
            utils_torch.AddLog("Useable GPU Num: %d"%GPUNum)
            report = "Useable GPU Memory: "
            for GPUIndex in range(GPUNum):
                report += "GPU%d: %.2fGB "%(GPUIndex, GPUUseableMemory[GPUIndex] * 1.0 / 1024 ** 3)
            utils_torch.AddLog(report)
        return 'cuda:%d'%(GPUWithLargestUseableMemoryIndex)
    except Exception:
        return "cuda:0"

def split_batch(data, batch_size): #data:(batch_size, image_size)
    sample_num = data.size(0)
    batch_sizes = [batch_size for _ in range(sample_num // batch_size)]
    if not sample_num % batch_size==0:
        batch_sizes.apend(sample_num % batch_size)
    return torch.split(data, section=batch_sizes, dim=0)

def cat_batch(dataloader): #data:(batch_num, batch_size, image_size)
    if not isinstance(dataloader, list):
        dataloader = list(dataloader)
    return torch.cat(dataloader, dim=0)

def ImportModule(ModulePath):
    try:
        return importlib.import_module(ModulePath)
    except Exception:
        return eval(ModulePath)

def import_file(file_from_sys_path):
    if not os.path.isfile(file_from_sys_path):
        raise Exception("%s is not a file."%file_from_sys_path)
    if file_from_sys_path.startswith("/"):
        raise Exception("import_file: file_from_sys_path must not be absolute path.")
    if file_from_sys_path.startswith("./"):
        module_path = file_from_sys_path.lstrip("./")
    module_path = module_path.replace("/", ".")
    return importlib.ImportModule(module_path)

def CopyDict(Dict):
    return dict(Dict)

def GetItemsFromDict(dict_, keys):
    items = []
    for name in keys:
        items.append(dict_[name])
    if len(items) == 1:
        return items[0]
    else:
        return tuple(items)   

def write_dict_info(dict_, save_path='./', save_name='dict info.txt'): # write readable dict info into file.
    values_remained = []
    with open(save_path+save_name, 'w') as f:
        for key in dict_.keys():
            value = dict_[value]
            if isinstance(value, str) or isinstance(value, int):
                f.write('%s: %s'%(str(key), str(value)))
            else:
                values_remained.append([key, value])

def GetNonLinearMethodModule(Name):
    if Name in ['relu']:
        return nn.ReLU()
    elif Name in ['tanh']:
        return nn.Tanh()
    elif Name in ['softplus']:
        return nn.Softplus()
    elif Name in ['sigmoid']:
        return nn.Sigmoid()
    else:
        raise Exception(Name)

def trunc_prefix(string, prefix):
    if(string[0:len(prefix)]==prefix):
        return string[len(prefix):len(string)]
    else:
        return string

def update_key(dict_0, dict_1, prefix='', strip=False, strip_only=True, exempt=[]):
    if not strip:
        for key in dict_1.keys():
            dict_0[prefix+key]=dict_1[key]
    else:
        for key in dict_1.keys():
            trunc_key=trunc_prefix(key, prefix)
            if strip_only:
                if(trunc_key!=key or key in exempt):
                    dict_0[trunc_key]=dict_1[key]
            else:
                dict_0[trunc_key]=dict_1[key]

def set_instance_attr(self, dict_, keys=None, exception=[]):
    if keys is None: # set all keys as instance variables.
        for key, value in dict_.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key not in exception:
                setattr(self, key, value)
    else: # set values of designated keys as instance variables.
        for key, value in dict_.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key in keys:
                if key not in exception:
                    setattr(self, key, value)

set_instance_variable = set_instance_attr

def set_dict_variable(dict_1, dict_0, keys=None, exception=['self']): # dict_1: target. dict_0: source.
    if keys is None: # set all keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key not in exception:
                dict_1[key] = value
    else: # set values of designated keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key in keys:
                if key not in exception:
                    dict_1[key] = value
        
def set_instance_variable_and_dict(self, dict_1, dict_0, keys=None, exception=['self']): # dict_0: source. dict_1: target dict. self: target class object.
    if keys is None: # set all keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key not in exception:
                dict_1[key] = value
                setattr(self, key, value)
    else: # set values of designated keys as instance variables.
        for key, value in dict_0.items(): # In python 3, use dict_.items(). In python 2, use dict_.iteritems()
            if key in keys:
                if key not in exception:
                    dict_1[key] = value
                    setattr(self, key, value)
                
def set_default_attr(self, key, value):
    if self.__dict__.get(key) is None:
        setattr(self, key, value)

set_dict_and_instance_variable = set_class_variable_and_dict = set_instance_variable_and_dict


def load_param(dict_, exception=[], default_exception=['kw', 'param', 'key', 'item'], use_default_exception=True):
    param = Param()
    for key, item in dict_.items():
        if key not in exception:
            if use_default_exception:
                if key not in default_exception:
                    setattr(param, key, item)
            else:
                setattr(param, key, item)
    return param

def print_dict(dict_):
    Str = ""
    
    for key, items in dict_.items():
        Str('%s=%s'%(str(key), str(items)), end=' ')
    print('\n')

def GetLastestModel(model_prefix, base_dir='./', is_dir=True):
    # search for directory or file of most recently saved models(model with biggest epoch index)
    if is_dir:
        max_epoch = None
        pattern = model_prefix+'(\d*)'
        dirs = os.listdir(base_dir)
        for dir_name in dirs:
            result = re.search(r''+pattern, dir_name)
            if result is not None:
                try:
                    epoch_num = int(result.group(1))
                except Exception:
                    print('error in matching model name.')
                    continue
                if(max_epoch is None):
                    max_epoch = epoch_num
                else:
                    if(max_epoch < epoch_num):
                        max_epoch = epoch_num
    if max_epoch is not None:
        return base_dir + model_prefix + str(max_epoch) + '/'
    else:
        return "error"

def standardize_suffix(suffix):
    pattern = re.compile(r'\.?(\w+)')
    result = pattern.match(suffix)
    if result is None:
        raise Exception('check_suffix: %s is illegal suffix.'%suffix)
    else:
        suffix = result.group(1)
    return suffix

def EnsureSuffix(name, suffix):
    if not suffix.startswith("."):
        suffix = "." + suffix
    if name.endswith(suffix):
        return suffix
    else:
        return name + suffix

def check_suffix(name, suffix=None, is_path=True):
    # check whether given file name has suffix. If true, check whether it's legal. If false, add given suffix to it.
    if suffix is not None:
        if isinstance(suffix, str):
            suffix = standardize_suffix(suffix)
        elif isinstance(suffix, list):
            for i, suf_ in enumerate(suffix):
                suffix[i] = standardize_suffix(suf_)
            if len(suffix)==0:
                suffix = None
        else:
            raise Exception('check_suffix: invalid suffix: %s'%(str(suffix)))      

    pattern = re.compile(r'(.*)\.(\w+)')
    result = pattern.match(name)
    if result is not None: # match succeeded
        name = result.group(1)
        suf = result.group(2)
        if suffix is None:
            return name + '.' + suf
        elif isinstance(suffix, str):
            if name==suffix:
                return name
            else:
                warnings.warn('check_suffix: %s is illegal suffix. replacing it with %s.'%(suf, suffix))
                return name + '.' + suffix
        elif isinstance(suffix, list):
            sig = False
            for suf_ in suffix:
                if suf==suf_:
                    sig = True
                    return name
            if not sig:
                warnings.warn('check_suffix: %s is illegal suffix. replacing it with %s.'%(suf, suffix[0]))
                return name + '.' + suffix[0]                
        else:
            raise Exception('check_suffix: invalid suffix: %s'%(str(suffix)))
    else: # fail to match
        if suffix is None:
            raise Exception('check_suffix: %s does not have suffix.'%name)
        else:
            if isinstance(suffix, str):
                suf_ = suffix
            elif isinstance(suffix, str):
                suf_ = suffix[0]
            else:
                raise Exception('check_suffix: invalid suffix: %s'%(str(suffix)))
            warnings.warn('check_suffix: no suffix found in %s. adding suffix %s.'%(name, suffix))            
            return name + '.' + suf_

def remove_suffix(name, suffix='.py', must_match=False):
    pattern = re.compile(r'(.*)%s'%suffix)
    result = pattern.match(name)
    if result is None:
        if must_match:
            raise Exception('%s does not have suffix %s'%(name, suffix))
        else:
            return name
    else:
        return result.group(1)

def scan_files(path, pattern, ignore_folder=True, raise_not_found_error=False):
    if not path.endswith('/'):
        path.append('/')
    files_path = os.listdir(path)
    matched_files = []
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    for file_name in files_path:
        #print(file_name)
        if pattern.match(file_name) is not None:
            if os.path.isdir(path + file_name):
                if ignore_folder:
                    matched_files.append(file_name)
                else:
                    warnings.warn('%s is a folder, and will be ignored.'%(path + file))
            else:
                matched_files.append(file_name)
    
    if raise_not_found_error:
        if len(matched_files)==0:
            raise Exception('scan_files: cannot find any files that match pattern %s'%pattern)

    return matched_files

def copy_files(file_list, SourceDir='./', TargetDir=None, sys_type='linux'):
    if not SourceDir.endswith('/'):
        SourceDir += '/'

    if not TargetDir.endswith('/'):
        TargetDir += '/'

    EnsurePath(TargetDir)

    '''
    if subpath is not None:
        if not subpath.endswith('/'):
             subpath += '/'
        path += subpath
    EnsurePath(path)
    '''
    #print(TargetDir)
    if sys_type in ['linux']:
        for file in file_list:
            file = file.lstrip('./')
            file = file.lstrip('/')
            #print(path)
            #print(file)
            #shutil.copy2(file, dest + file)
            #print(SourceDir + file)
            #print(TargetDir + file)
            EnsurePath(os.path.dirname(TargetDir + file))
            if os.path.exists(TargetDir + file):
                os.system('rm -r %s'%(TargetDir + file))
            #print('cp -r %s %s'%(file_path + file, path + file))
            os.system('cp -r %s %s'%(SourceDir + file, TargetDir + file))
    elif sys_type in ['windows']:
        # to be implemented 
        pass
    else:
        raise Exception('copy_files: Invalid sys_type: '%str(sys_type))


def TargetDir_module(path):
    path = path.lstrip('./')
    path = path.lstrip('/')
    if not path.endswith('/'):
        path += '/'
    path =  path.replace('/','.')
    return path


def GetAllMethodsOfModule(ModulePath):
    from inspect import getmembers, isfunction
    Module = ImportModule(ModulePath)
    return getmembers(Module, isfunction)

ListAllMethodsOfModule = GetAllMethodsOfModule

# GlobalParam = utils_torch.json.JsonObj2PyObj({
#     "Logger": None
# })

def RandomSelect(List, SelectNum):
    if isinstance(List, int):
        Num = List
        List = range(Num)
    else:
        Num = utils_torch.GetLength(List)

    if Num > SelectNum:
        return random.sample(List, SelectNum)
    else:
        return List

def GetLength(Obj):
    if utils_torch.IsIterable(Obj):
        return len(Obj)
    else:
        raise Exception()

import subprocess
def runcmd(command):
    ret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",timeout=1)
    if ret.returncode == 0:
        print("success:",ret)
    else:
        print("error:",ret)

def CalculateGitProjectTotalLines(Verbose=False):
    # runcmd(
    #     "git log  --pretty=tformat: --numstat | awk '{ add += $1; subs += $2; loc += $1 - $2 } END { printf \"added lines: %s, removed lines: %s, total lines: %s\n\", add, subs, loc }'"
    # )
    # GitCommand = 'git log  --pretty=tformat: --numstat | awk "{ add += $1; subs += $2; loc += $1 - $2 } END { printf "added lines: %s, removed lines: %s, total lines: %s\n", add, subs, loc }"'
    # report = os.system(GitCommand)
    # if Verbose:
    #     utils_torch.AddLog(report)
    # return report
    import os
    GitCommand = 'git log  --pretty=tformat: --numstat | awk \'{ add += $1; subs += $2; loc += $1 - $2 } END { printf "added lines: %s, removed lines: %s, total lines: %s\\n", add, subs, loc }\''
    report = os.system(GitCommand)

def GetDimensionNum(data):
    if isinstance(data, torch.Tensor):
        return len(list(data.size()))
    elif isinstance(data, np.ndarray):
        return len(data.shape)
    else:
        raise Exception(type(data))

def ToLowerStr(Str):
    return Str.lower()

def Str2File(Str, FilePath):
    with open(FilePath, "w") as file:
        file.write(Str)

def GetSavePathFromName(Name, Suffix=""):
    if not Suffix.startswith("."):
        Suffix = "." + Suffix
    FilePath = utils_torch.GetMainSaveDir() + Name + Suffix
    FilePath = utils_torch.RenameIfPathExists(FilePath)
    return FilePath

def Data2TextFile(data, Name=None, FilePath=None):
    if FilePath is None:
        FilePath = GetSavePathFromName(Name, Suffix=".txt")
    utils_torch.Str2File(str(data), FilePath)

def Float2StrDisplay(Float):
    if np.isinf(Float):
        return "inf"
    if np.isneginf(Float):
        return "-inf"
    if np.isnan(Float):
        return "NaN"

    if Float==0.0:
        return "0.0"

    Positive = Float < 0.0
    if not Positive:
        Float = - Float
        Sign = - 1.0
    else:
        Sign = 1.0

    Base, Exp = utils_torch.math.Float2BaseAndExponent(Float)
    TicksStr = []
    if 1 <= Exp <= 2:
        FloatStr = str(int(Float))
    elif Exp == 0:
        FloatStr = '%.1f'%Float
    elif Exp == -1:
        FloatStr = '%.2f'%Float
    elif Exp == -2:
        FloatStr = '%.3f'%Float
    else:
        FloatStr = '%.2e'%Float
    return FloatStr * Sign

def Floats2StrDisplay(Floats):
    Floats = ToNpArray(Floats)
    Base, Exp = utils_torch.math.FloatsBaseAndExponent(Floats)

def Floats2StrWithEqualLength(Floats):
    Floats = utils_torch.ToNpArray(Floats)
    Base, Exp = utils_torch.math.Floats2BaseAndExponent(Floats)
    # to be implemented

def MountObj(MountPath, Obj, **kw):
    ObjRoot = kw.get("ObjRoot")
    ObjCurrent = kw.get("ObjCurrent")

    MountPath = MountPath.replace("/&", "&")
    MountPath = MountPath.replace("&^", "ObjRoot.")
    MountPath = MountPath.replace("&", "ObjCurrent.")
    MountPath = MountPath.split(".")
    SetAttrs(eval(MountPath[0]), MountPath[1:], value=Obj)

def MountDictOnObj(Obj, Dict):
    Obj.__dict__.update(Dict)

ExternalMethods = utils_torch.PyObj()
def RegisterExternalMethods(Name, Method):
    setattr(ExternalMethods, Name, Method)

def Bytes2Str(Bytes, Format="utf-8"):
    return str(Bytes, encoding = "utf-8")

def Str2Bytes(Str, Format="utf-8"):
    return Str.decode(Format)


KB = 1024
MB = 1048576
GB = 1073741824
TB = 1099511627776


def ByteNum2Str(ByteNum):
    if ByteNum < KB:
        Str = "%d B"%ByteNum
    elif ByteNum < MB:
        Str = "%.3f KB"%(1.0 * ByteNum / KB)
    elif ByteNum < GB:
        Str = "%.3f MB"%(1.0 * ByteNum / MB)
    elif ByteNum < TB:
        Str = "%.3f GB"%(1.0 * ByteNum / GB)
    else:
        Str = "%.3f TB"%(1.0 * ByteNum / TB)
    return Str

def Unzip(Lists):
    return zip(*Lists)

def Zip(*Lists):
    return zip(*Lists)
