

import utils_torch
import utils_torch.module.AbstractModules as AbstractModules
from utils_torch.module.AbstractModules import AbstractModule, AbstractModuleWithParam, AbstractModuleWithoutParam

def BuildModuleFromType(Type):
    module = utils_torch.transform.BuildModuleIfIsLegalType(Type)
    if module is not None:
        return module
    
    module = utils_torch.loss.BuildModuleIfIsLegalType(Type)
    if module is not None:
        return module
    
    module = utils_torch.dataset.BuildModuleIfIsLegalType(Type)
    if module is not None:
        return module

    module = utils_torch.optimize.BuildModuleIfIsLegalType(Type)
    if module is not None:
        return module

    raise Exception()

def BuildModule(param, **kw):
    if hasattr(param, "ClassPath"):
        try:
            Class = utils_torch.parse.ParseClass(param.ClassPath)
            return Class(**kw)
        except Exception:
            utils_torch.AddWarning("Cannot parse ClassPath: %s"%param.ClassPath)
    # if param.Type in ['transform.RNNLIF']:
    #     print("aaa")
    module = utils_torch.transform.BuildModuleIfIsLegalType(param, **kw)
    if module is not None:
        return module
    
    module = utils_torch.loss.BuildModuleIfIsLegalType(param, **kw)
    if module is not None:
        return module
    
    module = utils_torch.dataset.BuildModuleIfIsLegalType(param, **kw)
    if module is not None:
        return module

    module = utils_torch.optimize.BuildModuleIfIsLegalType(param, **kw)
    if module is not None:
        return module

    module = BuildExternalModule(param, **kw)
    if module is not None:
        return module
    raise Exception()

ExternalModules = {}

def RegisterExternalModule(Type, Class):
    ExternalModules[Type] = Class

def BuildExternalModule(param, **kw):
    if isinstance(param, str):
        Type = param
    else:
        Type = param.Type
    if Type in ExternalModules:
        return ExternalModules[Type](**kw)
    else:
        return None