import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

class AbstractModule:
    def __init__(self, **kw):
        return
    def RegisterExternalMethod(self, Name, Method):
        if not callable(Method):
            Method = utils_torch.parse.ResolveStr(Method)
        setattr(self, Name, Method)
    def LoadParam(self, param):
        self.param = param
        return self
    def LoadData(self, data):
        self.data = data
        return self
    def LoadDataFromFile(self, FilePath):
        self.LoadData(
            utils_torch.file.DataFile2PyObj(FilePath)
        )
        return self
    def LoadDataFromDir(self, LoadDir):
        FilePath = LoadDir = self.param.FullName + ".data"
        self.LoadDataFromFile(self, FilePath)
        return self
    def SetTensorLocation(self, Location):
        cache = self.cache
        cache.TensorLocation = Location
        if hasattr(cache, "Tensors"):
            for ParamIndex in cache.Tensors:
                setattr(ParamIndex[0], ParamIndex[1], ParamIndex[2].to(Location).detach().requires_grad_(True))

        if hasattr(cache, "Modules"):
            for name, module in ListAttrsAndValues(cache.Modules):
                if hasattr(module, "SetTensorLocation"):
                    module.SetTensorLocation(Location)
                # else:
                #     if isinstance(module, nn.Module):
                #         utils_torch.AddWarning("%s is an instance of nn.Module, but has not implemented SetTensorLocation method."%name)
    
class AbstractModuleWithParam(AbstractModule):
    def __init__(self, **kw):
        return
    def BeforeBuild(self, IsLoad, ParseParam=True):
        cache = self.cache = utils_torch.EmptyPyObj()
        IsInit = not IsLoad
        cache.IsLoad = IsLoad
        cache.IsInit = IsInit
        cache.__object__ = self

        if IsInit:
            data = self.data = utils_torch.EmptyPyObj()
        else:
            assert hasattr(self, "data")

        assert hasattr(self, "param")
        param = self.param

        if not hasattr(param, "FullName"):
            if hasattr(self.__class__, "FullNameDefault"):
                param.FullName = self.__class__.FullNameDefault
            else:
                param.FullName = "DefaultFullName"

        if hasattr(param, "Modules"):
            self.Modules = cache.Modules = utils_torch.EmptyPyObj()
            param.Modules.SetResolveBase()
        if hasattr(param, "Dynamics"):
            self.Dynamics = cache.Dynamics = utils_torch.EmptyPyObj()
            param.Dynamics.SetResolveBase()

        if ParseParam:
            utils_torch.parse.ParsePyObjStatic(param, ObjCurrent=param)

    # def Build(self, IsLoad):
    #     cache = self.cache
    #     cache.IsLoad = IsLoad
    #     cache.IsInit = not IsLoad
    #     cache.__object__ = self

    #     self.Modules = cache.Modules
    #     self.Dynamics = cache.Dynamics

    #     utils_torch.parse.ParsePyObjStatic(self.param)
    def ParseRouters(self):
        GlobalParam = utils_torch.GetGlobalParam()
        param = self.param
        cache = self.cache
        for Name, RouterParam in ListAttrsAndValues(param.Dynamics, Exceptions=["__ResolveBase__", "__Entry__"]):
            if isinstance(RouterParam, str) and RouterParam in ["ClassMethod", "InternalMethod"]:
                setattr(cache.Dynamics, Name, getattr(self, Name))
                continue
            if cache.IsInit:
                utils_torch.router.ParseRouterStatic(RouterParam)
                setattr(RouterParam, "Name", param.FullName + "." + Name) # For Debug
            setattr(cache.Dynamics, Name, utils_torch.EmptyPyObj())
        
        ObjRefList = [
            cache.Modules, cache.Dynamics, cache,
            param, self, utils_torch.transform.operator,
        ]
        if hasattr(GlobalParam.cache, "AdditionalObjRefListForParseRouters"):
            ObjRefList += GlobalParam.cache.AdditionalObjRefListForParseRouters
        for Name, RouterParam in ListAttrsAndValues(param.Dynamics, Exceptions=["__ResolveBase__", "__Entry__"]):
            if isinstance(RouterParam, str) and RouterParam in ["ClassMethod", "InternalMethod"]:
                continue
            getattr(cache.Dynamics, Name).FromPyObj(
                utils_torch.router.ParseRouterDynamic(
                    RouterParam, 
                    ObjRefList = ObjRefList, ObjRoot = utils_torch.GetGlobalParam(),
                    InPlace=False
                )
            )
        return
    def InitModule(self, param=None, data=None, ClassPath=None, **kw):
        LoadDir = kw.get("LoadDir")
        FullName = kw.setdefault("FullName", "Unnamed")
        HasTensor = kw.setdefault("HasTensor", True)

        if param is None:
            param = utils_torch.EmptyPyObj()
        
        EnsureAttrs(param, "FullName", default=FullName)

        param.cache.__object__ = self
        if hasattr(param, "Modules"):
            param.Modules.SetResolveBase()
        if hasattr(param, "Dynamics"):
            param.Dynamics.SetResolveBase()

        if data is None:
            if LoadDir is not None:
                DataPath = LoadDir + param.FullName + ".data"
                if utils_torch.FileExists(DataPath):
                    data = utils_torch.json.DataFile2PyObj(DataPath)
                else:
                    data = utils_torch.EmptyPyObj()
            else:
                data = utils_torch.EmptyPyObj()

        cache = utils_torch.EmptyPyObj()
        if LoadDir is not None:
            cache.LoadDir = LoadDir
        else:
            cache.LoadDir = None
        if ClassPath is not None:
            param.ClassPath = ClassPath
        
        cache.Modules = utils_torch.EmptyPyObj()
        cache.Dynamics = utils_torch.EmptyPyObj()

        if HasTensor:
            cache.Tensors = []

        self.param = param
        self.data = data
        self.cache = cache
        self.Modules = cache.Modules
        self.Dynamics = cache.Dynamics

    def SetFullName(self, FullName):
        cache = self.cache
        param = self.param
        if FullName not in [""]:
            param.FullName = FullName
        if hasattr(cache, "Modules"):   
            for Name, Module in ListAttrsAndValues(cache.Modules):
                if hasattr(Module, "SetFullName"):
                    if FullName in [""]:
                        Module.SetFullName(Name)
                    else:
                        Module.SetFullName(FullName + "." + Name)

    def GetFullName(self):
        if hasattr(self, "param"):
            if hasattr(self.param, "FullName"):
                return self.param.FullName
        return None
    def DoInitTasks(self):
        param = self.param
        EnsureAttrs(param, "InitTasks", default=[])
        for Task in self.param.InitTasks:
            utils_torch.DoTask(Task, ObjCurrent=self.cache, ObjRoot=utils_torch.GetGlobalParam())
    def LoadFromParam(self):
        self.Build(IsLoad=True)
    def ParseParam(self, **kw):
        GlobalParam = utils_torch.GetGlobalParam()
        utils_torch.parse.ParsePyObjStatic(self.param, ObjCurrent=self.param, ObjRoot=GlobalParam)
    def ToFile(self, SaveName=None, SaveDir=None):
        param = self.param
        data = self.data
        if SaveName is None:
            SaveName = self.param.FullName
        if not data.IsEmpty():
            utils_torch.file.PyObj2DataFile(data,  SaveDir + SaveName + ".data")
            
        utils_torch.file.PyObj2JsonFile(param, SaveDir + SaveName + ".jsonc")
        return self
    def FromFile(self, SaveDir, SaveName, LoadParam=False, IsRoot=False):
        if IsRoot:
            LoadParam = True
        self.data  = utils_torch.file.DataFile2PyObj(SaveDir + SaveName + ".data")
        if LoadParam:
            self.param = utils_torch.file.JsonFile2PyObj(SaveDir + SaveName + ".jsonc")
        param = self.param
        self.cache = utils_torch.EmptyPyObj()
        if hasattr(param, "Modules"):
            for ModuleName, ModuleParam in param.Modules.Items():
                utils_torch.module.BuildModule(ModuleParam).FromFile(
                    SaveDir, SaveName + "." + ModuleName, LoadParam=False
                )
        return self
    def OverwriteParam(self, ParamPath, Value):
        SetAttrs(self.param, ParamPath, value=Value)
    def InitForNonModel(self, param=None, data=None, ClassPath=None, **kw):
        self.InitForModule(self, param, data, ClassPath, HasTensor=False, **kw)
        return
    def BuildModules(self, IsLoad=False, LoadDir=None):
        # initialize modules
        # for module in ListAttrs(param.modules):
        IsInit = not IsLoad
        
        param = self.param
        cache = self.cache
        for Name, ModuleParam in ListAttrsAndValues(param.Modules, Exceptions=["__ResolveBase__"]):
            ModuleParam.Name = Name
            ModuleParam.FullName = param.FullName + "." + Name

            if not HasAttrs(ModuleParam, "Type"):
                if HasAttrs(ModuleParam, "Name"):
                    SetAttrs(ModuleParam, "Type", GetAttrs(ModuleParam.Name))
                else:
                    raise Exception()
            if ModuleParam.Type in ["Internal", "External"]:
                continue
            module = utils_torch.module.BuildModule(ModuleParam)
            if IsInit:
                if hasattr(module, "LoadParam"):
                    module.LoadParam(ModuleParam)
            else:
                if hasattr(module, "LoadParam"):
                    module.LoadParam(ModuleParam)
                if hasattr(module, "LoadDataFromFile"):
                    module.LoadDataFromFile(LoadDir)
            # if isinstance(module, nn.Module) and isinstance(self, nn.Module):
            #     self.add_module(Name, Module)
            setattr(cache.Modules, Name, module)
    # def BuildModules(self):
    #     # initialize modules
    #     # for module in ListAttrs(param.modules):
    #     param = self.param
    #     cache = self.cache
    #     for Name, ModuleParam in ListAttrsAndValues(param.Modules, Exceptions=["__ResolveBase__"]):
    #         ModuleParam.Name = Name
    #         ModuleParam.FullName = param.FullName + "." + Name

    #         if not HasAttrs(ModuleParam, "Type"):
    #             if HasAttrs(ModuleParam, "Name"):
    #                 SetAttrs(ModuleParam, "Type", GetAttrs(ModuleParam.Name))
    #             else:
    #                 raise Exception()
    #         if ModuleParam.Type in ["Internal", "External"]:
    #             continue
    #         if cache.IsInit:
    #             module = utils_torch.module.BuildModule(ModuleParam)
    #             if hasattr(module, "LoadParam"):
    #                 module.LoadParam(ModuleParam)
    #         else:
    #             module = utils_torch.module.BuildModule(ModuleParam).LoadDataFromDir(cache.LoadDir)
    #         # if isinstance(module, nn.Module) and isinstance(self, nn.Module):
    #         #     self.add_module(Name, module)
    #         setattr(cache.Modules, Name, module)
class AbstractModuleWithoutParam(AbstractModule):
    def __init__(self):
        return
    def ToFile(self, FilePath):
        if not FilePath.endswith(".data"):
            FilePath += ".data"
        utils_torch.file.PyObj2DataFile(self.data, FilePath)
        return self
    def FromFile(self, FilePath):
        assert FilePath.endswith(".data")
        self.data = utils_torch.file.DataFile2PyObj(FilePath)
        return self
    def BeforeBuild(self, IsLoad=False):
        self.cache = utils_torch.EmptyPyObj()
        return

    