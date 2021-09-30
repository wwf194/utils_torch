
import utils_torch
import matplotlib as mpl
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
        self.PlotType = defaultdict("Unknown")
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
    def PlotAllLogs(self, Name, SaveDir=None):
        if Name not in self.log:
            raise Exception(Name)
        utils_torch.EnsureDir(SaveDir)
        for Name, Log in self.log.items():
            if self.PlotType[Name] in ["Unknown"]:
                try:
                    Xs = []
                    Ys = []
                    for Record in Log:
                        Xs.append(Record[2])
                        Ys.append(utils_torch.train.GetEpochFloat(Record[0], Record[1], self.BatchNum))
                    fig, ax = plt.subplots()
                    utils_torch.plot.PlotLineChart(ax, Xs, Ys, Title="%s-Epoch"%key, XLabel="Epoch", YLabel=key)
                    utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "%s.png")
                except Exception:
                    continue
            elif self.PlotType[Name] in ["DictItems"]:
                Xs = defaultdict(lambda:[])
                Ys = defaultdict(lambda:[])
                for Record in Log:
                    for key, value in Record[2].items():
                        Xs[key].append(value)
                        Ys[key].append(utils_torch.train.GetEpochFloat(Record[0], Record[1], self.BatchNum))
                PlotNum = len(Xs.keys())
                fig, axes = utils_torch.plot.CreateFigurePlt(PlotNum)
                for index, key in enumerate(Xs.keys()):
                    ax = utils_torch.plot.GetAx(axes, Index=index)
                    utils_torch.plot.PlotLineChart(ax, Xs[key], Ys[key], Title="%s-Epoch"%key, XLabel="Epoch", YLabel=key)
                plt.tight_layout()
                utils_torch.plot.SaveFigForPlt(SavePath=SaveDir + "%s.png")
            else:
                raise Exception()
