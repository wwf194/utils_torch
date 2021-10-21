import re
import sys
import torch

import utils_torch

def ReportPyTorchInfo():
    Report = ""
    if torch.cuda.is_available():
        Report += "Cuda is available"
    else:
        Report += "Cuda is unavailable"
    Report += "\n"
    Report += "Torch version:"+torch.__version__
    return Report

def GetSystemType():
    if re.match(r'win',sys.platform) is not None:
        SystemType = 'windows'
    elif re.match(r'linux',sys.platform) is not None:
        SystemType = 'linux'
    else:
        SystemType = 'unknown'
    return SystemType

def GetBytesInMemory(Obj):
    return sys.getsizeof(Obj)

def ReportMemoryOccupancy(Obj):
    ByteNum = GetBytesInMemory(Obj)
    return utils_torch.ByteNum2Str(Obj)