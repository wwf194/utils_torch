import torch
import torch.nn as nn
import torch.nn.functional as F
import utils_torch
from utils_torch.attr import *
import utils_torch.loss.classification as classification

from utils_torch.loss.L2Loss import L2Loss

from utils_torch.loss.classification import Probability2MostProbableIndex, LogAccuracyForSingleClassPrediction

ModuleList = [
    "L2Loss",
    "CrossEntropyLossForSingleClassPrediction",
    "MeanSquareError", "MSE",
]

def BuildModuleIfIsLegalType(param, **kw):
    if isinstance(param, str):
        Type = param
    else:
        Type = param.Type
    
    if IsLegalModuleType(Type):
        return BuildModule(param, **kw)
    else:
        return None
def IsLegalModuleType(Type):
    return Type in ModuleList

def BuildModule(param, **kw):
    if isinstance(param, str):
        Type = param
    else:
        Type = param.Type
    if Type in ["MeanSquareError", "MSE"]:
        Coefficient = param.Coefficient
        if Coefficient == 1.0:
            return F.mse_loss
        else:
            return lambda x, xTarget: Coefficient * F.mse_loss(x, xTarget)
    elif Type in ["CrossEntropyLossForSingleClassPrediction", "CrossEntropyLossForLabels"]:
        # By convention, softmax is included in the loss function.
        # Hard labels. Input: [SampleNum, ClassNum]. Target: Labels in shape of [SampleNum]
        Coefficient = param.Coefficient
        if Coefficient == 1.0:
            return F.cross_entropy
        else:
            return lambda Output, ProbabilitiesTarget:Coefficient * F.cross_entropy(Output, ProbabilitiesTarget)
    elif Type in ["CrossEntropyLoss", "CEL"]:
        # By convention, softmax is included in the loss function.
        # Soft labels. Input: [SampleNum, ClassNum]. Target: Probabilities in shape of [SampleNum, ClassNum]
        Coefficient = param.Coefficient
        if Coefficient == 1.0:
            return CrossEntropyLossForTargetProbability
        else:
            return lambda Output, ProbabilitiesTarget:Coefficient * CrossEntropyLossForTargetProbability(Output, ProbabilitiesTarget)
    elif Type in ["L2Loss"]:
        return L2Loss(**kw)
    else:
        raise Exception(Type)
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