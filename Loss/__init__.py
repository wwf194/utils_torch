import torch
import torch.nn as nn
import torch.nn.functional as F
import utils_torch
from utils_torch.attrs import *

from utils_torch.Loss.L2Loss import L2Loss

ModuleList = [
    "L2Loss",
    "CrossEntropyLossForSingleClassPrediction",
    "MeanSquareError", "MSE",
]

def IsLegalModuleType(Type):
    return Type in ModuleList
def BuildModule(param, **kw):
    if param.Type in ["MeanSquareError", "MSE"]:
        Coefficient = param.Coefficient
        if Coefficient == 1.0:
            return F.mse_loss
        else:
            return lambda x, xTarget: Coefficient * F.mse_loss(x, xTarget)
    elif param.Type in ["CrossEntropyLossForSingleClassPrediction", "CrossEntropyLossForLabels"]:
        # By convention, softmax is included in the loss function.
        # Hard labels. Input: [SampleNum, ClassNum]. Target: Labels in shape of [SampleNum]
        Coefficient = param.Coefficient
        if Coefficient == 1.0:
            return F.cross_entropy
        else:
            return lambda Output, ProbabilitiesTarget:Coefficient * F.cross_entropy(Output, ProbabilitiesTarget)
    elif param.Type in ["CrossEntropyLoss", "CEL"]:
        # By convention, softmax is included in the loss function.
        # Soft labels. Input: [SampleNum, ClassNum]. Target: Probabilities in shape of [SampleNum, ClassNum]
        Coefficient = param.Coefficient
        if Coefficient == 1.0:
            return CrossEntropyLossForTargetProbability
        else:
            return lambda Output, ProbabilitiesTarget:Coefficient * CrossEntropyLossForTargetProbability(Output, ProbabilitiesTarget)
    elif param.Type in ["L2Loss"]:
        return L2Loss(param, **kw)
    else:
        raise Exception(param.Type)
GetLossMethod = BuildModule

def CrossEntropyLossForTargetProbability(Output, ProbabilityTarget, Method='Average'):
    # Output: [SampleNum, OutputNum]
    # ProbabilitiesTarget: [SampleNum, OutputNum], must be positive, and sum to 1 on axis 1.
    LogProbabilities = -F.log_softmax(Output, dim=1) # [SampleNum, OutputNum]
    BatchSize = Output.shape[0]
    CrossEntropy = torch.sum(LogProbabilities * ProbabilityTarget, axis=1) # [SampleNum]
    if Method == 'Average':
        CrossEntropy = torch.mean(CrossEntropy)
    elif Method in ["Sum"]:
        CrossEntropy = torch.sum(CrossEntropy)
    else:
        raise Exception(Method)
    return CrossEntropy
ModuleList.append("CrossEntropyLossForTargetProbability")
ModuleList = set(ModuleList) # For faster indexing