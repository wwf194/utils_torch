import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

def GetLossMethod(param):
    if param.Type in ["MeanSquareError", "MSE"]:
        Coefficient = param.Coefficient
        if Coefficient==1.0:
            return F.mse_loss
        else:
            return lambda x, xTarget: Coefficient * F.mse_loss(x, xTarget)
    else:
        raise Exception(param.Type)

