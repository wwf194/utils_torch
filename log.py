
import utils_torch
from pydblite import Base

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