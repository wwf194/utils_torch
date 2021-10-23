import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.Modules.MLP import MLP
from utils_torch.Modules.SignalTrafficNodes import SerialReceiver
from utils_torch.Modules.SignalTrafficNodes import SerialSender
from utils_torch.Modules.SignalTrafficNodes import SignalHolder
from utils_torch.Modules.LambdaLayer import LambdaLayer
from utils_torch.Modules.RecurrentLIFLayer import RecurrentLIFLayer
from utils_torch.Modules.NoiseGenerator import NoiseGenerator
from utils_torch.Modules.Bias import Bias
import utils_torch.Modules.Operators as Operators
from utils_torch.Modules.SingleLayer import SingleLayer
from utils_torch.Modules.NonLinearLayer import NonLinearLayer
from utils_torch.Modules.LinearLayer import LinearLayer

import utils_torch
ModuleList = [
    "LinearLayer", "NonLinearLayer",
    "NoiseGenerator",
    "RecurrentLIFLayer",
    "LambdaLayer", "Lambda",
    "MLP",
    "SerialSender", "SerialReceiver", "SignalHolder",
    "NonLinear", # NonLinear Function
    "Bias",
]
def IsLegalModuleType(Type):
    return Type in ModuleList

def BuildModule(param, **kw):
    if param.Type in ["LinearLayer"]:
        return utils_torch.Modules.LinearLayer(param, **kw)
    elif param.Type in ["NonLinearLayer"]:
        return utils_torch.Modules.NonLinearLayer(param, **kw)
    elif param.Type in ["MLP", "MultiLayerPerceptron", "mlp"]:
        return utils_torch.Modules.MLP(param, **kw)
    elif param.Type in ["SerialReceiver"]:
        return utils_torch.Modules.SerialReceiver(param, **kw)
    elif param.Type in ["SerialSender"]:
        return utils_torch.Modules.SerialSender(param, **kw)
    elif param.Type in ["SignalHolder"]:
        return utils_torch.Modules.SignalHolder(param, **kw)
    elif param.Type in ["Lambda", "LambdaLayer"]:
        return utils_torch.Modules.LambdaLayer(param, **kw)
    elif param.Type in ["RecurrentLIFLayer"]:
        return utils_torch.Modules.RecurrentLIFLayer(param, **kw)
    elif param.Type in ["NoiseGenerator"]:
        return utils_torch.Modules.NoiseGenerator(param, **kw)
    elif param.Type in ["Bias"]:
        return utils_torch.Modules.Bias(param, **kw)
    elif param.Type in ["NonLinear"]:
        return GetNonLinearMethod(param, **kw)
    else:
        raise Exception(param.Type)

def GetNonLinearMethod(param, **kw):
    param = ParseNonLinearMethod(param)
    if param.Type in ["NonLinear"]:
        if hasattr(param, "Subtype"):
            Type = param.Subtype
    else:
        Type = param.Type

    if Type in ["relu", "ReLU"]:
        if param.Coefficient==1.0:
            return F.relu
        else:
            return lambda x:param.Coefficient * F.relu(x)
    elif Type in ["tanh", "Tanh"]:
        if param.Coefficient==1.0:
            return F.tanh
        else:
            return lambda x:param.Coefficient * F.tanh(x)       
    elif Type in ["sigmoid", "Sigmoid"]:
        if param.Coefficient==1.0:
            return F.tanh
        else:
            return lambda x:param.Coefficient * F.tanh(x)         
    else:
        raise Exception("GetNonLinearMethod: Invalid nonlinear function Type: %s"%param.Type)
GetActivationFunction = GetNonLinearMethod

def ParseNonLinearMethod(param):
    if isinstance(param, str):
        param = utils_torch.PyObj({
            "Type": param,
            "Coefficient": 1.0
        })
    elif isinstance(param, list):
        if len(param)==2:
            param = utils_torch.PyObj({
                "Type": param[0],
                "Coefficient": param[1]
            })
        else:
            # to be implemented
            pass
    elif isinstance(param, utils_torch.PyObj):
        if not hasattr(param, "Coefficient"):
            param.Coefficient = 1.0
    else:
        raise Exception("ParseNonLinearMethod: invalid param Type: %s"%type(param))
    return param

ModuleList = set(ModuleList)