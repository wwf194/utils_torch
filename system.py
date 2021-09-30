import re
import sys
import torch

def GetPyTorchInfo():
    if torch.cuda.is_available():
        print("Cuda is available")
    else:
        print("Cuda is unavailable")
    print("Torch version:"+torch.__version__)

def GetSystemType():
    if re.match(r'win',sys.platform) is not None:
        SystemType = 'windows'
    elif re.match(r'linux',sys.platform) is not None:
        SystemType = 'linux'
    else:
        SystemType = 'unknown'
    return SystemType

