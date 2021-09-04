import os
import utils_torch
def RemoveAllFiles(path, verbose=True):
    if not os.path.exists(path):
        raise Exception()
    if not os.path.isdir(path):
        raise Exception()
    for file in GetAllFiles(path):
        file_path = os.path.join(path, file)
        os.remove(file_path)
        utils_torch.AddLog("utils_pytorch: removed file: %s"%file_path)

def GetAllFiles(DirPath):
    if not os.path.exists(DirPath):
        raise Exception()
    if not os.path.isdir(DirPath):
        raise Exception()
    items = os.listdir(DirPath)
    Files = []
    for item in items:
        if os.path.isfile(os.path.join(DirPath, item)):
            Files.append(item)
    return Files

ListAllFiles = GetAllFiles

def GetAllDirs(DirPath):
    if not os.path.exists(DirPath):
        raise Exception()
    if not os.path.isdir(DirPath):
        raise Exception()
    items = os.listdir(DirPath)
    Dirs = []
    for item in items:
        if os.path.isdir(os.path.join(DirPath, item)):
            Dirs.append(item)
    return Dirs

ListAllDirs = GetAllDirs
