

import utils_torch
import utils_torch.module.AbstractModules as AbstractModules
from utils_torch.module.AbstractModules import \
    AbstractModule, AbstractTransformModule, \
    AbstractModuleWithTensor, \
    AbstractTransformModuleWithTensor, \
    AbstractModuleForEpochBatchTrain

def BuildModule(param, **kw):
    module = utils_torch.transform.BuildModuleIfIsLegalType(param, **kw)
    if module is not None:
        return
    
    module = utils_torch.loss.BuildModuleIfIsLegalType(param, **kw)
    if module is not None:
        return
    
    module = utils_torch.dataset.BuildModuleIfIsLegalType(param, **kw)
    if module is not None:
        return


    elif utils_torch.transform.Operators.IsLegalModuleType(param.Type):
        return utils_torch.transform.Operators.BuildModule(param, **kw)
    elif utils_torch.optimize.IsLegalModuleType(param.Type):
        return utils_torch.optimize.BuildModule(param, **kw)
    else:
        raise Exception()


