import argparse
parser = argparse.ArgumentParser()
parser.add_argument("task", nargs="?", default="ProcessTasks")
parser.add_argument("-IsDebug", default=True)
Args = parser.parse_args()

def AddSysPath():
    import os
    import sys
    sys.path.append(os.path.abs(".."))
AddSysPath()

import utils_torch

def main():
    if Args.task in ["CleanLog", "CleanLog", "cleanlog"]:
        CleanLog()
    elif Args.task in ["CleanFigure"]:
        CleanFigures()
    elif Args.task in ["TotalLines"]:
        utils_torch.CalculateGitProjectTotalLines()
    else:
        raise Exception("Inavlid Task: %s"%Args.task)

def CleanLog():
    import utils_torch
    utils_torch.file.RemoveAllFilesAndDirs("./log/")

def CleanFigures():
    import utils_torch
    utils_torch.file.RemoveMatchedFiles("./", r".*\.png")

if __name__=="__main__":
    main()