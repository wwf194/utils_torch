import utils_torch

import numpy as np
import matplotlib as mpl
import pandas as pd

from matplotlib import pyplot as plt
from pydblite import Base
from collections import defaultdict

import time
import logging
import json5
from inspect import getframeinfo, stack

import utils_torch
from utils_torch.attrs import *

class DataLog:
    def __init__(self, IsRoot=False):
        if IsRoot:
            self.tables = {}
        param = self.param = utils_torch.EmptyPyObj()
        cache = self.cache = utils_torch.EmptyPyObj()
        cache.LocalColumn = {}
        param.LocalColumnNames = cache.LocalColumn.keys()
        self.HasParent = False
        return
    def SetParent(self, log, prefix=""):
        self.parent = log
        self.parentPrefix = prefix
        self.HasParent = True
        self.IsRoot = False
        return self
    def GetParent(self):
        return self.parent
    def SetParentPrefix(self, prefix):
        if not self.HasParent:
            raise Exception()
        self.parentPrefix = prefix
    def SetLocal(self, Name, Value):
        param = self.param
        cache = self.cache
        cache.LocalColumn[Name] = Value
        param.LocalColumnNames = cache.LocalColumn.keys()
        return self
    def CreateTable(self, TableName, ColumnNames, SavePath):
        param = self.param
        if self.HasParent:
            table = self.parent.CreateTable(self.parentPrefix + TableName, [*ColumnNames, *param.LocalColumnNames], SavePath)
        else:
            if hasattr(self.tables, TableName):
                utils_torch.AddWarning("Table with name: %s already exists."%TableName)
            utils_torch.EnsureFileDir(SavePath)
            utils_torch.EnsureFileDir(SavePath)
            table = Base(SavePath)
            table.create(*ColumnNames)
            self.tables[TableName] = table
        return table     
    def GetTable(self, TableName):
        table = self.tables.get(TableName)
        # if table is None:
        #     raise Exception("No such table: %s"%TableName)
        return table
    def HasTable(self, TableName):
        return self.tables.get(TableName) is None
    def CreateIndex(self, TableName, IndexColumn):
        table = self.GetTable(TableName)
        table.create_index(IndexColumn)
    def AddRecord(self, TableName, ColumnValues, AddLocalColumn=True):
        param = self.param
        cache = self.cache
        if self.HasParent:
            if AddLocalColumn:
                ColumnValues.update(cache.LocalColumn)
            self.parent.AddRecord(self.parentPrefix + TableName, ColumnValues)
        else:
            table = self.GetTable(TableName)
            if table is None:
                #raise Exception(TableName)
                table = self.CreateTable(TableName, [*ColumnValues.keys(), *param.LocalColumnNames], 
                    SavePath=utils_torch.GetMainSaveDir() + "data/" + "%s.pdl"%TableName)
            if AddLocalColumn:
                table.insert(**ColumnValues, **self.cache.LocalColumn)
            else:
                table.insert(**ColumnValues, **self.cache.LocalColumn)
            table.commit()

def LogList2EpochsFloat(Log, **kw):
    # for _Log in Log:
    #     EpochIndices.append(_Log[0])
    #     BatchIndices.append(_Log[1])
    if ("EpochFloat" in Log) and len(Log["EpochFloat"])==len(Log["Epoch"]):
        pass
    else:
        Log["EpochFloat"] = utils_torch.train.EpochBatchIndices2EpochsFloat(Log["Epoch"], Log["Batch"], **kw)
    return Log["EpochFloat"]

def LogDict2EpochsFloat(Log, **kw):
    return utils_torch.train.EpochBatchIndices2EpochsFloat(Log["Epoch"], Log["Batch"], **kw)

def PlotLogList(Name, Log, SaveDir=None, **kw):
    EpochsFloat = LogList2EpochsFloat(Log, BatchNum=kw["BatchNum"])
    Ys = Log["Value"]
    fig, ax = plt.subplots()
    utils_torch.plot.PlotLineChart(ax, EpochsFloat, Ys, Title="%s-Epoch"%Name, XLabel="Epoch", YLabel=Name)
    utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "%s-Epoch.png"%Name)
    utils_torch.files.Table2TextFile(
        {
            "Epoch": EpochsFloat,
            Name: Ys,
        },
        SavePath=SaveDir + "%s-Epoch.txt"%Name
    )

class LogForEpochBatchTrain:
    def __init__(self, param=None, **kw):
        utils_torch.module.InitForNonModel(self, param, ClassPath="utils_torch.train.LogForEpochBatchTrain", **kw)
        self.InitFromParam(IsLoad=False)
    def InitFromParam(self, IsLoad=False):
        utils_torch.module.InitFromParamForModule(self, IsLoad)
        param = self.param
        data = self.data
        cache = self.cache
        data.log = defaultdict(lambda:[])
        #self.IsPlotable = defaultdict(lambda:True)
        data.logType = defaultdict(lambda:"Unknown")
        self.GetLog = self.GetLogByName
        self.AddLog = self.AddLogList
        self.Get = self.GetLog
    def UpdateEpoch(self, EpochIndex):
        cache = self.cache
        cache.EpochIndex = EpochIndex
    def UpdateBatch(self, BatchIndex):
        cache = self.cache
        cache.BatchIndex = BatchIndex
    def AddLogList(self, Name, Value, Type=None):
        data = self.data
        cache =self.cache
        if not Name in data.log:
            data.log[Name] = {
                "Epoch":[],
                "Batch":[],
                "Value":[]
            }
            if Type is not None:
                data.logType[Name] = Type
        log = data.log[Name]
        log["Epoch"].append(cache.EpochIndex)
        log["Batch"].append(cache.BatchIndex),
        log["Value"].append(Value)
    def AddLogDict(self, Name, Dict, Type=None):
        data = self.data
        cache = self.cache
        if not Name in data.log:
            data.log[Name] = defaultdict(lambda:[])
            if Type is not None:
                data.logType[Name] = Type
        Log = data.log[Name]
        for key, value in Dict.items():
            Log[key].append(value)
        Log["Epoch"].append(cache.EpochIndex)
        Log["Batch"].append(cache.BatchIndex)
    def AddLogCache(self, Name, Data, Type="Cache"):
        cache = self.cache
        data = self.data
        data.logType[Name] = Type
        data.log[Name] = {
            "Epoch": cache.EpochIndex,
            "Batch":cache.BatchIndex,
            "Value":Data
        }
    def RegisterLog(self, Name, Type="List"):
        data = self.data
        if Type in ["List"]:
            data.log[Name] = []
        elif Type in ["Dict"]:
            data.log[Name] = {}
        else:
            raise Exception(Type)
    def SetPlotType(self, Name, Type):
        self.PlotType[Name] = Type

    def SetLocal(self, Name, Value):
        setattr(self, Name, Value)
    def SetLogType(self, Name, Value):
        data = self.data
        if not Name in data.log:
            raise Exception()
        data.logType[Name] = Value
    def PlotLogOfGivenType(self, Type, PlotType="LineChart", SaveDir=None):
        utils_torch.EnsureDir(SaveDir)
        data = self.data
        for Name, Log in data.log.items():
            if not data.logType[Name] in [Type]:
                continue
            if PlotType in ["LineChart"]:
                self.PlotLogList(Name, Log, SaveDir)
            elif PlotType in ["Statistics"]:
                self.PlotLogDictStatistics(Name, Log, SaveDir)
            else:
                raise Exception(PlotType)
    def Log2EpochsFloat(self, Log, **kw):
        kw["BatchNum"] = self.BatchNum
        return utils_torch.train.EpochBatchIndices2EpochsFloat(Log["Epoch"], Log["Batch"], **kw)
    def PlotLogDict(self, Name, Log, SaveDir=None):
        utils_torch.EnsureDir(SaveDir)
        LogNum = len(Log.keys()[0])
        PlotNum = len(Log.keys() - 2) # Exclude Epoch, Batch
        fig, axes = utils_torch.plot.CreateFigurePlt(PlotNum)
        Xs = self.GetEpochsFloatFromLogDict(Log)
        for index, Key in enumerate(Log.keys()):
            Ys = Log[Key]
            ax = utils_torch.plot.GetAx(axes, Index=index)
            utils_torch.plot.PlotLineChart(ax, Xs, Ys, Title="%s-Epoch"%Key, XLabel="Epoch", YLabel=Key)
        plt.tight_layout()
        utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "%s.png"%Name)
        utils_torch.files.Table2TextFileDict(Log, SavePath=SaveDir + "%s-Epoch"%Name)
    def GetLogValueByName(self, Name):
        Log = self.GetLogByName(Name)
        if isinstance(Log, dict) and "Value" in Log:
            return Log["Value"]
        else:
            return Log
    def GetLogByName(self, Name):
        data = self.data
        if not Name in data.log:
            #raise Exception(Name)
            utils_torch.AddWarning("No such log: %s"%Name)
            return None
        return data.log[Name]
    def GetCacheByName(self, Name):
        data = self.data
        if not Name in data.log:
            #raise Exception(Name)
            utils_torch.AddWarning("No such log: %s"%Name)
            return None
        return data.log[Name]["Value"]
    def GetLogOfType(self, Type):
        data = self.data
        Logs = {}
        for Name, Log in data.log.items():
            if data.logType[Name] == Type:
                Logs[Name] = Log
        return Logs
    def PlotAllLogs(self, SaveDir=None):
        utils_torch.EnsureDir(SaveDir)
        data = self.data
        for Name, Log in data.log.items():
            if isinstance(Log, dict):
                self.PlotLogDict(self, Name, Log, SaveDir)
            elif isinstance(Log, list):
                self.PlotLogList(self, Name, Log, SaveDir)
            else:
                continue
utils_torch.module.SetEpochBatchMethodForModule(LogForEpochBatchTrain)

class Log:
    def __init__(self, Name, **kw):
        self.log = _CreateLog(Name, **kw)
    def AddLog(self, log, TimeStamp=True, FilePath=True, RelativeFilePath=True, LineNum=True, StackIndex=1, **kw):
        Caller = getframeinfo(stack()[StackIndex][0])
        if TimeStamp:
            log = "[%s]%s"%(utils_torch.system.GetTime(), log)
        if FilePath:
            if RelativeFilePath:
                log = "%s File \"%s\""%(log, utils_torch.files.GetRelativePath(Caller.filename, "."))
            else:
                log = "%s File \"%s\""%(log, Caller.filename)
        if LineNum:
            log = "%s, line %d"%(log, Caller.lineno)
        self.log.debug(log)

    def AddWarning(self, log, TimeStamp=True, File=True, LineNum=True, StackIndex=1, **kw):
        Caller = getframeinfo(stack()[StackIndex][0])
        if TimeStamp:
            log = "[%s][WARNING]%s"%(utils_torch.system.GetTime(), log)
        if File:
            log = "%s File \"%s\""%(log, Caller.filename)
        if LineNum:
            log = "%s, line %d"%(log, Caller.lineno)
        self.log.debug(log)

    def AddError(self, log, TimeStamp=True, **kw):
        if TimeStamp:
            self.log.error("[%s][ERROR]%s"%(utils_torch.system.GetTime(), log))
        else:
            self.log.error("%s"%log)
    def Save(self):
        return

def ParseLog(log, **kw):
    if log is None:
        log = GetLogGlobal()
    elif isinstance(log, str):
        log = GetLog(log, **kw)
    else:
        return log
    return log

def AddLog(Str, log=None, *args, **kw):
    ParseLog(log, **kw).AddLog(Str, *args, StackIndex=2, **kw)

def AddWarning(Str, log=None, *args, **kw):
    ParseLog(log, **kw).AddWarning(Str, *args, StackIndex=2, **kw)

def AddError(Str, log=None, *args, **kw):
    ParseLog(log, **kw).AddError(Str, *args, StackIndex=2, **kw)

def AddLog2GlobalParam(Name, **kw):
    import utils_torch
    setattr(utils_torch.GlobalParam.log, Name, CreateLog(Name, **kw))

def CreateLog(Name, **kw):
    return Log(Name, **kw)

def _CreateLog(Name, SaveDir=None, **kw):
    if SaveDir is None:
        SaveDir = utils_torch.GetMainSaveDir()
    utils_torch.EnsureDir(SaveDir)
    HandlerList = ["File", "Console"]
    if kw.get("FileOnly"):
        HandlerList = ["File"]

    # 输出到file
    log = logging.Logger(Name)
    log.setLevel(logging.DEBUG)
    log.HandlerList = HandlerList

    for HandlerType in HandlerList:
        if HandlerType in ["Console"]:
            # 输出到console
            ConsoleHandler = logging.StreamHandler()
            ConsoleHandler.setLevel(logging.DEBUG) # 指定被处理的信息级别为最低级DEBUG，低于level级别的信息将被忽略
            log.addHandler(ConsoleHandler)
        elif HandlerType in ["File"]:
            FileHandler = logging.FileHandler(SaveDir + "%s.txt"%(Name), mode='w', encoding='utf-8')  # 不拆分日志文件，a指追加模式,w为覆盖模式
            FileHandler.setLevel(logging.DEBUG)     
            log.addHandler(FileHandler)
        else:
            raise Exception(HandlerType)

    HandlerNum = len(HandlerList)
    if len(HandlerList)==0:
        raise Exception(HandlerNum)

    return log

def SetLogGlobal(GlobalParam):
    GlobalParam.log.Global = CreateLog('Global')

def SetLog(Name, log):
    setattr(utils_torch.GlobalParam.log, Name, log)

def GetLogGlobal():
    return utils_torch.GlobalParam.log.Global

def SetGlobalParam(GlobalParam):
    utils_torch.GlobalParam = GlobalParam

def GetGlobalParam():
    return utils_torch.GlobalParam

def GetDatasetDir(Type):
    GlobalParam = utils_torch.GetGlobalParam()
    Attrs = "config.Dataset.%s.Dir"%Type
    if not HasAttrs(GlobalParam, Attrs):
        raise Exception()
    else:
        return GetAttrs(GlobalParam, Attrs)
    
def SetMainSaveDir(SaveDir=None, Name=None, GlobalParam=None, Method="FromIndex"):
    if GlobalParam is None:
        GlobalParam = utils_torch.GetGlobalParam()
    if SaveDir is None:
        if Method in ["FromTime", "FromTimeStamp"]:
            SaveDir = "./log/%s-%s/"%(Name, utils_torch.system.GetTime("%Y-%m-%d-%H:%M:%S"))
        elif Method in ["FromIndex"]:
            SaveDir = utils_torch.RenameDirIfExists("./log/%s/"%Name)
        else:
            raise Exception(Method)
    utils_torch.EnsureDir(SaveDir)
    #print("[%s]Using Main Save Dir: %s"%(utils_torch.system.GetTime(),SaveDir))
    #utils_torch,AddLog("[%s]Using Main Save Dir: %s"%(utils_torch.system.GetTime(),SaveDir))
    SetAttrs(utils_torch.GetGlobalParam(), "SaveDir.Main", value=SaveDir)
    return SaveDir

def SetSubSaveDir(SaveDir=None, Name="Experiment", GlobalParam=None):
    if GlobalParam is None:
        GlobalParam = utils_torch.GetGlobalParam()
    SetAttrs(GlobalParam, "SaveDir" + "." + Name, value=SaveDir)
    return SaveDir

def GetSubSaveDir(Type):
    if not hasattr(utils_torch.GetGlobalParam().SaveDir, Type):
        setattr(utils_torch.GetGlobalParam().SaveDir, Type, utils_torch.GetMainSaveDir() + Type + "/")
    return getattr(utils_torch.GetGlobalParam().SaveDir, Type)        

def GetMainSaveDir(GlobalParam=None):
    if GlobalParam is None:
        GlobalParam = utils_torch.GetGlobalParam()
    return GlobalParam.SaveDir.Main

def SetSubSaveDirEpochBatch(Name, EpochIndex, BatchIndex, BatchInternalIndex=None, GlobalParam=None):
    if GlobalParam is None:
        GlobalParam = utils_torch.GetGlobalParam()
    if BatchInternalIndex is None:
        DirName = "Epoch%d-Batch%d"%(EpochIndex, BatchIndex)
    else:
        DirName = "Epoch%d-Batch%d-No%d"%(EpochIndex, BatchIndex, BatchInternalIndex)
    SaveDir = utils_torch.GetMainSaveDir(GlobalParam) + Name + "/" +  DirName + "/"
    SetSubSaveDir(Name, SaveDir)
    SetAttrs(GlobalParam, "SaveDirs" + "." + Name + "." + DirName, SaveDir)
    return SaveDir
def GetSubSaveDirEpochBatch(Name, EpochIndex, BatchIndex, BatchInternalIndex=None, GlobalParam=None):
    if BatchInternalIndex is None:
        DirName = "Epoch%d-Batch%d"%(EpochIndex, BatchIndex)
    else:
        DirName = "Epoch%d-Batch%d-No%d"%(EpochIndex, BatchIndex, BatchInternalIndex)
    # if HasAttrs(GlobalParam, "SaveDirs" + "." + Name + "." + DirName):
    #     return GetAttrs(GlobalParam, "SaveDirs" + "." + Name + "." + DirName)
    # else: # As a guess
    #     utils_torch.GetMainSaveDir(GlobalParam) + Name + "/" +  DirName + "/"
    return utils_torch.GetMainSaveDir(GlobalParam) + Name + "/" +  DirName + "/"

def GetAllSubSaveDirsEpochBatch(Name, SaveDir=None, GlobalParam=None):
    if GlobalParam is None:
        GlobalParam = utils_torch.GetGlobalParam()
    if SaveDir is None:
        SaveDir = utils_torch.GetMainSaveDir(GlobalParam)
    SaveDirs = utils_torch.files.ListAllDirs(SaveDir + Name + "/")
    SaveDirNum = len(SaveDirs)
    if SaveDirNum == 0:
        raise Exception(SaveDirNum)
    for Index, SaveDir in enumerate(SaveDirs):
        SaveDirs[Index] = utils_torch.GetMainSaveDir(GlobalParam) + Name + "/" + SaveDir # SaveDir already ends with "/"
    return SaveDirs

def GetDataLog():
    return utils_torch.GlobalParam.log.Data

def GetLog(Name, CreateIfNone=True, **kw):
    if not hasattr(utils_torch.GlobalParam.log, Name):
        if CreateIfNone:
            utils_torch.AddLog(Name)
        else:
            raise Exception()
    return getattr(utils_torch.GlobalParam.log, Name)

