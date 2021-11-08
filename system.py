import re
import sys
import torch
import time
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

def ReportTorchInfo(): # print info about training environment, global variables, etc.
    return torch.pytorch_info()

import subprocess
def RunPythonScript(FilePath, Args):
    ArgsList = ["python", FilePath, *Args]
    ArgsListStr = []
    for Arg in ArgsList:
        ArgsListStr.append(str(Arg))
    subprocess.call(ArgsListStr)
RunPythonFile = RunPythonScript

def GetTime(format="%Y-%m-%d %H:%M:%S", verbose=False):
    TimeStr = time.strftime(format, time.localtime()) # Time display style: 2016-03-20 11:45:39
    if verbose:
        print(TimeStr)
    return TimeStr

import dateutil
def GetTimeDifferenceFromStr(TimeStr1, TimeStr2):
    Time1 = dateutil.parser.parse(TimeStr1)
    Time2 = dateutil.parser.parse(TimeStr2)

    TimeDiffSeconds = (Time2 - Time1).total_seconds()
    TimeDiffSeconds = round(TimeDiffSeconds)

    _Second = TimeDiffSeconds % 60
    Minute = TimeDiffSeconds // 60
    _Minute = Minute % 60
    Hour = Minute // 60
    TimeDiffStr = "%d:%02d:%02d"%(Hour, _Minute, _Second)
    return TimeDiffStr