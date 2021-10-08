import enum
import utils_torch

import numpy as np
import matplotlib as mpl
import pandas as pd

from matplotlib import pyplot as plt
from pydblite import Base
from collections import defaultdict

import os
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
                    SavePath=utils_torch.GetSaveDir() + "data/" + "%s.pdl"%TableName)
            if AddLocalColumn:
                table.insert(**ColumnValues, **self.cache.LocalColumn)
            else:
                table.insert(**ColumnValues, **self.cache.LocalColumn)
            table.commit()
    
class LoggerForEpochBatchTrain:
    def __init__(self):
        self.log = defaultdict(lambda:[])
        self.IsPlotable = defaultdict(lambda:True)
        self.logType = defaultdict(lambda:"Unknown")
        self.PlotType = defaultdict(lambda:"Unknown")
        self.AddLog = self.AddLogList
        self.Get = self.GetLog
    def UpdateEpoch(self, EpochIndex):
        self.EpochIndex = EpochIndex
    def UpdateBatch(self, BatchIndex):
        self.BatchIndex = BatchIndex
    def AddLogList(self, Name, Value, Type=None):
        if not Name in self.log:
            self.log[Name] = []
            if Type is not None:
                self.logType[Name] = Type
        self.log[Name].append([self.EpochIndex, self.BatchIndex, Value])
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
        self.log[Name] = [self.EpochIndex, self.BatchIndex, data]
    def RegisterLog(self, Name, Type="List"):
        if Type in ["List"]:
            self.log[Name] = []
        elif Type in ["Dict"]:
            self.log[Name] = {}
        else:
            raise Exception(Type)
    def AddLogStatistics(self, Name, data, Type):
        data = utils_torch.ToNpArray(data)
        _Name = Name + ".statistics"
        if _Name not in self.log:
            self.log[_Name] = defaultdict(lambda:[])
            self.logType[_Name] = Type
        statistics = utils_torch.math.NpStatistics(data, ReturnType="Dict")
        self.AddLogDict(_Name, statistics)
    def SetPlotType(self, Name, Type):
        self.PlotType[Name] = Type
    def SetEpochNum(self, EpochNum):
        self.EpochNum = EpochNum
    def SetBatchNum(self, BatchNum):
        self.BatchNum = BatchNum
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
    def GetEpochsFloatFromLogDict(self, Log):
        LogNum = len(Log.keys()[0])
        Epochs = []
        for Index in range(LogNum):
            Epochs.append(utils_torch.train.GetEpochFloat(Log["Epoch"][Index], Log["Batch"][Index], self.BatchNum))
        return Epochs

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
    def PlotLogDictStatistics(self, Name, Log, SaveDir=None):
        utils_torch.EnsureDir(SaveDir)
        Epochs = self.GetEpochsFloatFromLogDict(Log)
        fig, ax = plt.subplots()
        utils_torch.plot.PlotMeanAndStd(
            Name, Log, Title="%s-statistics"%Name, XLabel="Epoch", YLabel=Name,
        )
        plt.tight_layout()
        utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "%s-statistics.png"%Name)
        utils_torch.files.Table2TextFileDict(Log, SavePath=SaveDir + "%s.statistics-Epoch.txt"%Name)
    def PlotLogList(self, Name, Log, SaveDir=None):
        Xs = []
        Ys = []
        for Record in Log:
            Xs.append(utils_torch.train.GetEpochFloat(Record[0], Record[1], self.BatchNum))
            Ys.append(Record[2])
        fig, ax = plt.subplots()
        utils_torch.plot.PlotLineChart(ax, Xs, Ys, Title="%s-Epoch"%Name, XLabel="Epoch", YLabel=Name)
        utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "%s-Epoch.png"%Name)
        utils_torch.files.Table2TextFile(
            {
                "Epoch": Xs,
                Name: Ys,
            },
            SavePath=SaveDir + "%s-Epoch.txt"%Name
        )
    def GetLog(self, Name):
        if not Name in self.log:
            raise Exception(Name)
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
    def __init__(self, Name):
        self.logger = _CreateLogger(Name)
    def AddLog(self, log, TimeStamp=True, File=True, LineNum=True, StackIndex=1):
        Caller = getframeinfo(stack()[StackIndex][0])
        if TimeStamp:
            log = "[%s]%s"%(utils_torch.GetTime(), log)
        if File:
            log = "%s File \"%s\""%(log, Caller.filename)
        if LineNum:
            log = "%s, line %d"%(log, Caller.lineno)
        self.logger.debug(log)

    def AddWarning(self, log, TimeStamp=True, File=True, LineNum=True, StackIndex=1):
        Caller = getframeinfo(stack()[StackIndex][0])
        if TimeStamp:
            log = "[%s][WARNING]%s"%(utils_torch.GetTime(), log)
        if File:
            log = "%s File \"%s\""%(log, Caller.filename)
        if LineNum:
            log = "%s, line %d"%(log, Caller.lineno)
        self.logger.debug(log)

    def AddError(self, log, TimeStamp=True):
        if TimeStamp:
            self.logger.error("[%s][ERROR]%s"%(utils_torch.GetTime(), log))
        else:
            self.logger.error("%s"%log)

def ParseLogger(logger):
    if logger is None:
        logger = GetLoggerGlobal()
    elif isinstance(logger, str):
        logger = GetLogger(logger)
    else:
        raise Exception()
    return logger

def AddLog(log, logger=None, *args, **kw):
    ParseLogger(logger).AddLog(log, *args, StackIndex=2, **kw)

def AddWarning(log, logger=None, *args, **kw):
    ParseLogger(logger).AddWarning(log, *args, StackIndex=2, **kw)

def AddError(log, logger=None, *args, **kw):
    ParseLogger(logger).AddError(log, *args, StackIndex=2, **kw)

def GetLogger(Name, CreateIfNone=True):
    if not hasattr(utils_torch.ArgsGlobal.logger, Name):
        if CreateIfNone:
            utils_torch.AddLogger(Name)
        else:
            raise Exception()
    return getattr(utils_torch.ArgsGlobal.logger, Name)

def CreateLogger(Name):
    return Logger(Name)

def _CreateLogger(Name, SaveDir=None):
    if SaveDir is None:
        SaveDir = utils_torch.GetSaveDir()
    
    # 输出到console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG) # 指定被处理的信息级别为最低级DEBUG，低于level级别的信息将被忽略
    utils_torch.EnsureDir(SaveDir)

    # 输出到file
    file_handler = logging.FileHandler(SaveDir + "%s.txt"%(Name), mode='w', encoding='utf-8')  # 不拆分日志文件，a指追加模式,w为覆盖模式
    file_handler.setLevel(logging.DEBUG)            
    logger = logging.Logger(Name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def SetLoggerGlobal(ArgsGlobal):
    ArgsGlobal.logger.Global = CreateLogger('Global')

def SetLogger(Name, logger):
    setattr(utils_torch.ArgsGlobal.logger, Name, logger)

def GetLoggerGlobal():
    return utils_torch.ArgsGlobal.logger.Global

def AddLogger(Name):
    import utils_torch
    setattr(utils_torch.ArgsGlobal.logger, Name, CreateLogger(Name))

def SetArgsGlobal(ArgsGlobal):
    utils_torch.ArgsGlobal = ArgsGlobal

def GetArgsGlobal():
    return utils_torch.ArgsGlobal

def SetSubSaveDir(SaveDir, Type, ArgsGlobal):
    SetAttrs(ArgsGlobal, "SaveDir" + "." + Type, value=SaveDir)
    utils_torch.EnsureDir(SaveDir)

def SetSaveDir(SaveDir=None, Type="Main", ArgsGlobal=None):
    if ArgsGlobal is None:
        ArgsGlobal = utils_torch.GetArgsGlobal()
    if Type in ["Main"]:
        SaveDir = "./log/Experiment-%s/"%(utils_torch.GetTime("%Y-%m-%d-%H:%M:%S"))
        utils_torch.EnsureDir(SaveDir)
        SetAttrs(utils_torch.GetArgsGlobal(), "SaveDir.Main", value=SaveDir)
    else:
        SetSubSaveDir(SaveDir, Type, ArgsGlobal)
def GetSaveDir(Type="Main"):
    if not hasattr(utils_torch.GetArgsGlobal().SaveDir, Type):
        setattr(utils_torch.GetArgsGlobal().SaveDir, Type, utils_torch.GetSaveDir() + Type + "/")
    return getattr(utils_torch.GetArgsGlobal().SaveDir, Type)        

def GetSaveDirForModel():
    return utils_torch.SetArgsGlobal.SaveDir.Model

def GetDataLogger():
    return utils_torch.ArgsGlobal.log.Data

def GetTime(format="%Y-%m-%d %H:%M:%S", verbose=False):
    TimeStr = time.strftime(format, time.localtime()) # Time display style: 2016-03-20 11:45:39
    if verbose:
        print(TimeStr)
    return TimeStr
