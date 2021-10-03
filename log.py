import enum
import utils_torch

import numpy as np
import matplotlib as mpl
import pandas as pd

from matplotlib import pyplot as plt
from pydblite import Base
from collections import defaultdict


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
        self.log[Name] = data
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
            utils_torch.plot.PlotLineChart(ax, Xs, Ys, Title="%s-Epoch"%Key, XLabel="Epoch", YLabel=key)
        plt.tight_layout()
        utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "%s.png"%Name)
        utils_torch.files.Table2TextFileDict(Log, SavePath="%s-Epoch"%Name)
    def PlotLogDictStatistics(self, Name, Log, SaveDir=None):
        utils_torch.EnsureDir(SaveDir)
        Epochs = self.GetEpochsFloatFromLogDict(Log)
        fig, ax = plt.subplots()
        utils_torch.plot.PlotMeanAndStd(
            Name, Log, Title="%s-statistics"%Name, XLabel="Epoch", YLabel=Name,
        )
        plt.tight_layout()
        utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "%s-statistics.png"%Name)
        utils_torch.files.Table2TextFileDict(Log, SavePath="%s.statistics-Epoch"%Name)
    def PlotLogList(self, Name, Log, SaveDir=None):
        Xs = []
        Ys = []
        for Record in Log:
            Xs.append(utils_torch.train.GetEpochFloat(Record[0], Record[1], self.BatchNum))
            Ys.append(Record[2])
        fig, ax = plt.subplots()
        utils_torch.plot.PlotLineChart(ax, Xs, Ys, Title="%s-Epoch"%Name, XLabel="Epoch", YLabel=Name)
        utils_torch.plot.SaveFigForPlt(SavePath="%s-Epoch.png"%Name)
        utils_torch.files.Table2TextFile(
            {
                "Epoch": Xs,
                Name: Ys,
            },
            SavePath=SaveDir + "%s-Epoch.txt"%Name
        )
    def GetLogCache(self, Name):
        if not Name in self:
            raise Exception(Name)
        return self.log[Name]
    def PlotAllLogs(self, SaveDir=None):
        utils_torch.EnsureDir(SaveDir)
        for Name, Log in self.log.items():
            if isinstance(Log, dict):
                self.PlotLogDict(self, Name, Log, SaveDir)
            elif isinstance(Log, list):
                self.PlotLogList(self, Name, Log, SaveDir)
            else:
                conti
                raise Exception()
            # if self.PlotType[Name] in ["Unknown"]:
            #     example = Log[0][2]
            #     if isinstance(example, np.ndarray):
            #         if not (len(example.shape)==1 and example.shape[0]==1 or len(example.shape)==0):
            #             continue
            # elif self.PlotType[Name] in ["DictItems"]:
            #     XsDict = defaultdict(lambda:[])
            #     YsDict = defaultdict(lambda:[])
            #     for Record in Log:
            #         for key, value in Record[2].items():
            #             XsDict[key].append(value)
            #             YsDict[key].append(utils_torch.train.GetEpochFloat(Record[0], Record[1], self.BatchNum))
            #     PlotNum = len(Xs.keys())
            #     fig, axes = utils_torch.plot.CreateFigurePlt(PlotNum)
            #     for index, key in enumerate(Xs.keys()):
            #         Xs, Ys = XsDict[key], YsDict[key]
            #         ax = utils_torch.plot.GetAx(axes, Index=index)
            #         utils_torch.plot.PlotLineChart(ax, Xs, Ys, Title="%s-Epoch"%key, XLabel="Epoch", YLabel=key)
            #     plt.tight_layout()
            #     utils_torch.log.Table2TextFile(
            #         {
            #             Name: Xs, 
            #             "Epoch": Ys,
            #         },
            #         SavePath=SaveDir + "%s-Epoch.txt"%Name
            #     )
            #     utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "%s.png")
            # else:
            #     raise Exception()