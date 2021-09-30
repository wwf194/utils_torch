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
        self.logDict = defaultdict(lambda:{})
        self.IsPlotable = defaultdict(lambda:True)
        self.PlotType = defaultdict(lambda:"Unknown")
    def UpdateEpoch(self, EpochIndex):
        self.EpochIndex = EpochIndex
    def UpdateBatch(self, BatchIndex):
        self.BatchIndex = BatchIndex
    def AddLog(self, Name, Value):
        self.log[Name].append([self.EpochIndex, self.BatchIndex, Value])
    def AddLog2Dict(self, Name, Dict):
        for Key, Value in Dict.items():
            self.logDict[Name + "." + Key].append([self.EpochIndex, self.BatchIndex, Value])
    def SetPlotType(self, Name, Type):
        self.PlotType[Name] = Type
    def SetEpochNum(self, EpochNum):
        self.EpochNum = EpochNum
    def SetBatchNum(self, BatchNum):
        self.BatchNum = BatchNum
    def SetLocal(self, Name, Value):
        setattr(self, Name, Value)
    def PlotAllLogs(self, SaveDir=None):
        # if Name not in self.log:
        #     raise Exception(Name)
        utils_torch.EnsureDir(SaveDir)
        for Name, Log in self.log.items():
            if self.PlotType[Name] in ["Unknown"]:
                example = Log[0][2]
                if isinstance(example, np.ndarray):
                    if not (len(example.shape)==1 and example.shape[0]==1 or len(example.shape)==0):
                        continue
                # try:
                Xs = []
                Ys = []
                for Record in Log:
                    Xs.append(utils_torch.train.GetEpochFloat(Record[0], Record[1], self.BatchNum))
                    Ys.append(Record[2])
                fig, ax = plt.subplots()
                utils_torch.plot.PlotLineChart(ax, Xs, Ys, Title="%s-Epoch"%Name, XLabel="Epoch", YLabel=Name)
                utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "%s-Epoch.png"%Name)
                utils_torch.log.Columns2File(Xs, Ys, Names=["Epoch", Name], SavePath=SaveDir + "%s-Epoch.txt"%Name)
            elif self.PlotType[Name] in ["DictItems"]:
                XsDict = defaultdict(lambda:[])
                YsDict = defaultdict(lambda:[])
                for Record in Log:
                    for key, value in Record[2].items():
                        XsDict[key].append(value)
                        YsDict[key].append(utils_torch.train.GetEpochFloat(Record[0], Record[1], self.BatchNum))
                PlotNum = len(Xs.keys())
                fig, axes = utils_torch.plot.CreateFigurePlt(PlotNum)
                for index, key in enumerate(Xs.keys()):
                    Xs, Ys = XsDict[key], YsDict[key]
                    ax = utils_torch.plot.GetAx(axes, Index=index)
                    utils_torch.plot.PlotLineChart(ax, Xs, Ys, Title="%s-Epoch"%key, XLabel="Epoch", YLabel=key)
                plt.tight_layout()
                utils_torch.log.Columns2File(Xs, Ys, Names=["Epoch", Name], SavePath=SaveDir + "%s-Epoch.txt"%Name)
                utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "%s.png")
            else:
                raise Exception()

def Columns2File(*Columns, **kw):
    # ColNum = len(Columns)
    # Str = " ".join(kw["Names"])
    # Str += "\n"
    # for RowIndex in range(len(Columns[0])):
    #     for ColIndex in range(ColNum):
    #         Str += str(Columns[ColIndex][RowIndex])
    #         Str += " "
    #     Str += "\n"
    # utils_torch.Str2File(Str, kw["SavePath"])

    Names = kw["Names"]
    Dict = {}
    for Index, Column in enumerate(Columns):
        Dict[Names[Index]] = Column
    utils_torch.Str2File(pd.DataFrame(Dict).to_string(), kw["SavePath"])

def Floats2StrWithEqualLength(Floats):
    np.log10()