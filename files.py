import os
import re
from typing import Match
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

def RemoveAllFilesAndDirs(path, verbose=True):
    if not os.path.exists(path):
        raise Exception()
    if not os.path.isdir(path):
        raise Exception()
    Files, Dirs= GetAllFilesAndDirs(path)
    for FileName in Files:
        FilePath = os.path.join(path, FileName)
        os.remove(FilePath)
        utils_torch.AddLog("utils_torch: removed file: %s"%FilePath)
    for DirName in Dirs:
        DirPath = os.path.join(path, DirName)
        #os.removedirs(DirPath) # Cannot delete subfolders
        import shutil
        shutil.rmtree(DirPath)
        utils_torch.AddLog("utils_torch: removed directory: %s"%DirPath)

def IsDir(DirPath):
    return os.path.isdir(DirPath)

def IsFile(FilePath):
    return os.path.isfile(FilePath)

def RemoveMatchedFiles(DirPath, Patterns):
    if not os.path.isdir(DirPath):
        raise Exception()
    if not DirPath.endswith("/"):
        DirPath += "/"
    if not isinstance(Patterns, list):
        Patterns = [Patterns]
    for Pattern in Patterns:
        FileNames = ListAllFiles(DirPath)
        for FileName in FileNames:
            MatchResult = re.match(Pattern, FileName)
            if MatchResult is not None:
                FilePath = os.path.join(DirPath, FileName)
                os.remove(FilePath)
                utils_torch.AddLog("utils_torch: removed file: %s"%FilePath)

def GetAllFilesAndDirs(DirPath):
    if not os.path.exists(DirPath):
        raise Exception()
    if not os.path.isdir(DirPath):
        raise Exception()
    items = os.listdir(DirPath)
    Files, Dirs = [], []
    for item in items:
        if os.path.isfile(os.path.join(DirPath, item)):
            Files.append(item)
        elif os.path.isdir(os.path.join(DirPath, item)):
            Dirs.append(item)
    return Files, Dirs

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

def ExistsFile(FilePath):
    return os.path.isfile(FilePath)

def Path2AbsolutePath(Path):
    return os.path.abspath(Path)

def EnsureDirectory(DirPath):
    #DirPathAbs = Path2AbsolutePath(DirPath)
    if os.path.exists(DirPath):
        if not os.path.isdir(DirPath):
            raise Exception("%s Already Exists but Is NOT a Directory."%DirPath)
    else:
        if not DirPath.endswith("/"):
            DirPath += "/"
        os.makedirs(DirPath)

EnsureDir = EnsureDirectory
EnsureFolder = EnsureDirectory

def EnsureFileDirectory(FilePath):
    if FilePath.endswith("/"):
        raise Exception()
    FilePath = Path2AbsolutePath(FilePath)
    FileDir = os.path.dirname(FilePath)
    EnsureDir(FileDir)

EnsureFileDir = EnsureFileDirectory

EnsureFolder = EnsureDir

def EnsurePath(path, isFolder=False): # check if given path exists. if not, create it.
    if isFolder: # caller of this function makes sure that path is a directory/folder.
        if not path.endswith('/'): # folder
            utils_torch.AddWarning('%s is a folder, and should ends with /.'%path)
            path += '/'
            #print(path)
            #input()
        if not os.path.exists(path):
            os.makedirs(path)
    else: # path can either be a directory or a folder. If path exists, then it is what it is (file or folder). If not, depend on whether it ends with '/'.
        if os.path.exists(path): # path exists
            if os.path.isdir(path):
                if not path.endswith('/'): # folder
                    path += '/'     
            elif os.path.isfile(path):
                raise Exception('file already exists: %s'%str(path))
            else:
                raise Exception('special file already exists: %s'%str(path))
        else: # path does not exists
            if path.endswith('/'): # path is a folder
                path_strip = path.rstrip('/')
            else:
                path_strip = path
            if os.path.exists(path_strip): # folder with same name exists
                raise Exception('EnsurePath: homonymous file exists.')
            else:
                if not os.path.exists(path_strip):
                    os.makedirs(path_strip)
                    #os.mkdir(path) # os.mkdir does not support creating multi-level folders.
                #filepath, filename = os.path.split(path)
    return path

ListAllDirs = GetAllDirs

def JoinPath(path_0, path_1):
    if not path_0.endswith('/'):
        path_0 += '/'
    if path_1.startswith('./'):
        path_1 = path_1.lstrip('./')
    if path_1.startswith('/'):
        raise Exception('join_path: path_1 is a absolute path: %s'%path_1)
    return path_0 + path_1

def CopyFolder(SourceDir, TargetDir, exceptions=[], verbose=True):
    '''
    if args.path is not None:
        path = args.path
    else:
        path = '/data4/wangweifan/backup/'
    '''
    #EnsurePath(SourceDir)
    EnsurePath(TargetDir)
    
    for i in range(len(exceptions)):
        exceptions[i] = os.path.abspath(exceptions[i])
        if os.path.isdir(exceptions[i]):
            exceptions[i] += '/'

    SourceDir = os.path.abspath(SourceDir)
    TargetDir = os.path.abspath(TargetDir)

    if not SourceDir.endswith('/'):
        SourceDir += '/'
    if not TargetDir.endswith('/'):
        TargetDir += '/'

    if verbose:
        print('Copying folder from %s to %s. Exceptions: %s'%(SourceDir, TargetDir, exceptions))

    if SourceDir + '/' in exceptions:
        utils_torch.AddWarning('CopyFolder: neglected the entire root path. nothing will be copied')
        if verbose:
            print('neglected')
    else:
        _CopyFolder(SourceDir, TargetDir, subpath='', exceptions=exceptions)

def _CopyFolder(SourceDir, TargetDir, subpath='', exceptions=[], verbose=True):
    #EnsurePath(SourceDir + subpath)
    EnsurePath(TargetDir + subpath)
    items = os.listdir(SourceDir + subpath)
    for item in items:
        #print(TargetDir + subpath + item)
        path = SourceDir + subpath + item
        if os.path.isfile(path): # is a file
            if path + '/' in exceptions:
                if verbose:
                    print('neglected file: %s'%path)
            else:
                if os.path.exists(TargetDir + subpath + item):
                    md5_source = Getmd5(SourceDir + subpath + item)
                    md5_target = Getmd5(TargetDir + subpath + item)
                    if md5_target==md5_source: # same file
                        #print('same file')
                        continue
                    else:
                        #print('different file')
                        os.system('rm -r "%s"'%(TargetDir + subpath + item))
                        os.system('cp -r "%s" "%s"'%(SourceDir + subpath + item, TargetDir + subpath + item))     
                else:
                    os.system('cp -r "%s" "%s"'%(SourceDir + subpath + item, TargetDir + subpath + item))
        elif os.path.isdir(path): # is a folder.
            if path + '/' in exceptions:
                if verbose:
                    print('neglected folder: %s'%(path + '/'))
            else:
                _CopyFolder(SourceDir, TargetDir, subpath + item + '/', verbose=verbose)
        else:
            utils_torch.AddWarning('%s is neither a file nor a path.')

def ExistsPath(Path):
    return os.path.exists(Path)

def ParseNameSuffix(FilePath):
    if FilePath.endswith("/"):
        raise Exception()
    MatchResult = re.match(r"(.*)\.(.*)", FilePath)
    if MatchResult is None:
        return FilePath, ""
    else:
        return MatchResult.group(1), MatchResult.group(2)

def RenameFileIfPathExists(FilePath):
    if FilePath.endswith("/"):
        raise Exception()

    FileName, Suffix = ParseNameSuffix(FilePath)

    if ExistsPath(FilePath):
        MatchResult = re.match(r"(.*)-(\d+)", FileName)
        if MatchResult is None:
            os.rename(FilePath, FileName + "-0" + "." + Suffix)
            FileNameOrigin = FileName
            Index = 1
        else:
            FileNameOrigin = MatchResult.group(1)
            Index = int(MatchResult.group(2))
        while True:
            FilePath = FileNameOrigin + "-%d"%Index + "." + Suffix
            if not ExistsPath(FilePath):
                return FilePath
            Index += 1
    else:
        return FilePath



