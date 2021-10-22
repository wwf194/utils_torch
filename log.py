import enum
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

class DataLogger:
    def __init__(self, IsRoot=False):
        if IsRoot:
            self.tables = {}
        param = self.param = utils_torch.EmptyPyObj()
        cache = self.cache = utils_torch.EmptyPyObj()
        cache.LocalColumn = {}
        param.LocalColumnNames = cache.LocalColumn.keys()
        self.HasParent = False
        return
    def SetParent(self, logger, prefix=""):
        self.parent = logger
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

def ListLog2EpochsFloat(Log, **kw):
    if isinstance(Log, dict):
        Log = list(Log.values())[0]
    return utils_torch.train.EpochBatchIndices2EpochsFloat(Log["Epoch"], Log["Batch"], **kw)

def PlotLogList(Name, Log, SaveDir=None, **kw):
    EpochsFloat = ListLog2EpochsFloat(Log, BatchNum=kw["BatchNum"])
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

class LoggerForEpochBatchTrain:
    def __init__(self):
        self.log = defaultdict(lambda:[])
        self.IsPlotable = defaultdict(lambda:True)
        self.logType = defaultdict(lambda:"Unknown")
        self.GetLog = self.GetLogByName
        self.AddLog = self.AddLogList
        self.Get = self.GetLog
    def UpdateEpoch(self, EpochIndex):
        self.EpochIndex = EpochIndex
    def UpdateBatch(self, BatchIndex):
        self.BatchIndex = BatchIndex
    def AddLogList(self, Name, Value, Type=None):
        if not Name in self.log:
            self.log[Name] = {
                "Epoch":[],
                "Batch":[],
                "Value":[]
            }
            if Type is not None:
                self.logType[Name] = Type
        #self.log[Name].append([self.EpochIndex, self.BatchIndex, Value])
        log = self.log[Name]
        log["Epoch"].append(self.EpochIndex)
        log["Batch"].append(self.BatchIndex),
        log["Value"].append(Value)
    def AddLogDict(self, Name, Dict, Type=None):
        if not Name in self.log:
            self.log[Name] = defaultdict(lambda:[])
            if Type is not None:
                self.logType[Name] = Type
        Log = self.log[Name]
        for key, value in Dict.items():
            Log[key].append(value)
        Log["Epoch"].append(self.EpochIndex)
        Log["Batch"].append(self.BatchIndex)
    def AddLogCache(self, Name, data, Type="Cache"):
        self.logType[Name] = Type
        self.log[Name] = {
            "Epoch":self.EpochIndex,
            "Batch":self.BatchIndex,
            "Value":data
        }
    def RegisterLog(self, Name, Type="List"):
        if Type in ["List"]:
            self.log[Name] = []
        elif Type in ["Dict"]:
            self.log[Name] = {}
        else:
            raise Exception(Type)
    def SetPlotType(self, Name, Type):
        self.PlotType[Name] = Type
    def NotifyEpochNum(self, EpochNum):
        self.EpochNum = EpochNum
    def NotifyBatchNum(self, BatchNum):
        self.BatchNum = BatchNum
    def NotifyEpochIndex(self, EpochIndex):
        self.EpochIndex = EpochIndex
    def NotifyBatchIndex(self, BatchIndex):
        self.BatchIndex = BatchIndex
    def SetLocal(self, Name, Value):
        setattr(self, Name, Value)
    def SetLogType(self, Name, Value):
        if not Name in self.log:
            raise Exception()
        self.logType[Name] = Value
    def PlotLogOfGivenType(self, Type, PlotType="LineChart", SaveDir=None):
        utils_torch.EnsureDir(SaveDir)
        for Name, Log in self.log.items():
            if not self.logType[Name] in [Type]:
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
    def GetLogByName(self, Name):
        if not Name in self.log:
            #raise Exception(Name)
            utils_torch.AddWarning("No such log: %s"%Name)
            return None
        return self.log[Name]
    def GetLogOfType(self, Type):
        Logs = {}
        for Name, Log in self.log.items():
            if self.logType[Name] in [Type]:
                Logs[Name] = Log
        return Logs
    def PlotAllLogs(self, SaveDir=None):
        utils_torch.EnsureDir(SaveDir)
        for Name, Log in self.log.items():
            if isinstance(Log, dict):
                self.PlotLogDict(self, Name, Log, SaveDir)
            elif isinstance(Log, list):
                self.PlotLogList(self, Name, Log, SaveDir)
            else:
                continue
    
class Logger:
    def __init__(self, Name, **kw):
        self.logger = _CreateLogger(Name, **kw)
    def AddLog(self, log, TimeStamp=True, File=True, LineNum=True, StackIndex=1, **kw):
        Caller = getframeinfo(stack()[StackIndex][0])
        if TimeStamp:
            log = "[%s]%s"%(utils_torch.GetTime(), log)
        if File:
            log = "%s File \"%s\""%(log, Caller.filename)
        if LineNum:
            log = "%s, line %d"%(log, Caller.lineno)
        self.logger.debug(log)

    def AddWarning(self, log, TimeStamp=True, File=True, LineNum=True, StackIndex=1, **kw):
        Caller = getframeinfo(stack()[StackIndex][0])
        if TimeStamp:
            log = "[%s][WARNING]%s"%(utils_torch.GetTime(), log)
        if File:
            log = "%s File \"%s\""%(log, Caller.filename)
        if LineNum:
            log = "%s, line %d"%(log, Caller.lineno)
        self.logger.debug(log)

    def AddError(self, log, TimeStamp=True, **kw):
        if TimeStamp:
            self.logger.error("[%s][ERROR]%s"%(utils_torch.GetTime(), log))
        else:
            self.logger.error("%s"%log)
    def Save(self):
        return

def ParseLogger(logger, **kw):
    if logger is None:
        logger = GetLoggerGlobal()
    elif isinstance(logger, str):
        logger = GetLogger(logger, **kw)
    else:
        raise Exception()
    return logger

def AddLog(log, logger=None, *args, **kw):
    ParseLogger(logger, **kw).AddLog(log, *args, StackIndex=2, **kw)

def AddWarning(log, logger=None, *args, **kw):
    ParseLogger(logger, **kw).AddWarning(log, *args, StackIndex=2, **kw)

def AddError(log, logger=None, *args, **kw):
    ParseLogger(logger, **kw).AddError(log, *args, StackIndex=2, **kw)

def AddLogger(Name, **kw):
    import utils_torch
    setattr(utils_torch.GlobalParam.log, Name, CreateLogger(Name, **kw))

def CreateLogger(Name, **kw):
    return Logger(Name, **kw)

def _CreateLogger(Name, SaveDir=None, **kw):
    if SaveDir is None:
        SaveDir = utils_torch.GetMainSaveDir()
    utils_torch.EnsureDir(SaveDir)
    HandlerList = ["File", "Console"]
    if kw.get("FileOnly"):
        HandlerList = ["File"]

    # 输出到file
    logger = logging.Logger(Name)
    logger.setLevel(logging.DEBUG)
    logger.HandlerList = HandlerList

    for HandlerType in HandlerList:
        if HandlerType in ["Console"]:
            # 输出到console
            ConsoleHandler = logging.StreamHandler()
            ConsoleHandler.setLevel(logging.DEBUG) # 指定被处理的信息级别为最低级DEBUG，低于level级别的信息将被忽略
            logger.addHandler(ConsoleHandler)
        elif HandlerType in ["File"]:
            FileHandler = logging.FileHandler(SaveDir + "%s.txt"%(Name), mode='w', encoding='utf-8')  # 不拆分日志文件，a指追加模式,w为覆盖模式
            FileHandler.setLevel(logging.DEBUG)     
            logger.addHandler(FileHandler)
        else:
            raise Exception(HandlerType)

    HandlerNum = len(HandlerList)
    if len(HandlerList)==0:
        raise Exception(HandlerNum)

    return logger

def SetLoggerGlobal(GlobalParam):
    GlobalParam.log.Global = CreateLogger('Global')

def SetLogger(Name, logger):
    setattr(utils_torch.GlobalParam.log, Name, logger)

def GetLoggerGlobal():
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

def SetMainSaveDir(SaveDir=None, Name=None, GlobalParam=None):
    if GlobalParam is None:
        GlobalParam = utils_torch.GetGlobalParam()
    if SaveDir is None:
        SaveDir = "./log/%s-%s/"%(Name, utils_torch.GetTime("%Y-%m-%d-%H:%M:%S"))

    utils_torch.EnsureDir(SaveDir)
    SetAttrs(utils_torch.GetGlobalParam(), "SaveDir.Main", value=SaveDir)

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

def GetAllSubSaveDirsEpochBatch(Name, GlobalParam=None):
    if GlobalParam is None:
        GlobalParam = utils_torch.GetGlobalParam()
    SaveDirs = utils_torch.files.ListAllDirs(utils_torch.GetMainSaveDir(GlobalParam) + Name + "/")
    SaveDirNum = len(SaveDirs)
    if SaveDirNum == 0:
        raise Exception(SaveDirNum)
    for Index, SaveDir in enumerate(SaveDirs):
        SaveDirs[Index] = utils_torch.GetMainSaveDir(GlobalParam) + Name + "/" + SaveDir # SaveDir already ends with "/"
    return SaveDirs

def GetDataLogger():
    return utils_torch.GlobalParam.log.Data

def GetLogger(Name, CreateIfNone=True, **kw):
    if not hasattr(utils_torch.GlobalParam.log, Name):
        if CreateIfNone:
            utils_torch.AddLogger(Name)
        else:
            raise Exception()
    return getattr(utils_torch.GlobalParam.log, Name)

def GetTime(format="%Y-%m-%d %H:%M:%S", verbose=False):
    TimeStr = time.strftime(format, time.localtime()) # Time display style: 2016-03-20 11:45:39
    if verbose:
        print(TimeStr)
    return TimeStr