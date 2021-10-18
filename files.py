import os
import re
import pandas as pd

import utils_torch
from utils_torch.attrs import *

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

def ListFiles(DirPath):
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

ListAllFiles = GetAllFiles = ListFiles

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
FileExists = ExistsFile

def CheckFileExists(FilePath):
    if not utils_torch.ExistsFile(FilePath):
        raise Exception("%s does not exist."%FilePath)

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

def RenameIfPathExists(FilePath):
    if FilePath.endswith("/"):
        raise Exception()

    FileName, Suffix = ParseNameSuffix(FilePath)

    Sig = True
    MatchResult = re.match(r"^(.*)-(\d+)$", FileName)
    if MatchResult is None:
        if ExistsPath(FilePath):
            os.rename(FilePath, FileName + "-0" + "." + Suffix)
            FileNameOrigin = FileName
            Index = 1
        elif ExistsPath(FileName + "-0" + "." + Suffix):
            FileNameOrigin = FileName
            Index = 1
        else:
            Sig = False
    else:
        FileNameOrigin = MatchResult.group(1)
        Index = int(MatchResult.group(2))
    if Sig:
        while True:
            FilePath = FileNameOrigin + "-%d"%Index + "." + Suffix
            if not ExistsPath(FilePath):
                return FilePath
            Index += 1
    else:
        return FilePath

def Table2TextFileDict(Dict, SavePath):
    utils_torch.Str2File(pd.DataFrame(Dict).to_string(), SavePath)   
Table2TextFile = Table2TextFileDict

def Table2TextFileColumns(*Columns, **kw):
    # ColNum = len(Columns)
    # Str = " ".join(kw["Names"])
    # Str += "\n"
    # for RowIndex in range(len(Columns[0])):
    #     for ColIndex in range(ColNum):
    #         Str += str(Columns[ColIndex][RowIndex])
    #         Str += " "
    #     Str += "\n"
    # utils_torch.Str2File(Str, kw["SavePath"])
    Names = kw["Names"]
    Dict = {}
    for Index, Column in enumerate(Columns):
        Dict[Names[Index]] = Column
    Table2TextFileDict(Dict, kw["SavePath"])

def LoadParamFromFile(Args, **kw):
    if isinstance(Args, dict):
        _LoadParamFromFile(utils_torch.json.JsonObj2PyObj(Args), **kw)
    elif isinstance(Args, list) or utils_torch.IsListLikePyObj(Args):
        for Arg in Args:
            _LoadParamFromFile(Arg, **kw)
    elif utils_torch.IsDictLikePyObj(Args):
        _LoadParamFromFile(Args, **kw)
    else:
        raise Exception()

def _LoadParamFromFile(Args, **kw):
    FilePathList = utils_torch.ToList(Args.FilePath)
    MountPathList = utils_torch.ToList(Args.MountPath)
    for MountPath, FilePath in zip(MountPathList, FilePathList):
        Obj = utils_torch.json.JsonFile2PyObj(FilePath)
        if not isinstance(Obj, list):
            EnsureAttrs(Args, "SetResolveBase", default=True)
            if Args.SetResolveBase:
                setattr(Obj, "__ResolveBase__", True)
        utils_torch.MountObj(MountPath, Obj, **kw)
    return

def cal_path_from_main(path_rel=None, path_start=None, path_main=None):
    # path_rel: file path relevant to path_start
    if path_main is None:
        path_main = sys.path[0]
    if path_start is None:
        path_start = path_main
        warnings.warn('cal_path_from_main: path_start is None. using default: %s'%path_main)
    path_start = os.path.abspath(path_start)
    path_main = os.path.abspath(path_main)
    if os.path.isfile(path_main):
        path_main = os.path.dirname(path_main)
    if not path_main.endswith('/'):
        path_main += '/' # necessary for os.path.relpath to calculate correctly
    if os.path.isfile(path_start):
        path_start = os.path.dirname(path_start)
    #path_start_rel = os.path.relpath(path_start, start=path_main)

    if path_rel.startswith('./'):
        path_rel.lstrip('./')
    elif path_rel.startswith('/'):
        raise Exception('path_rel: %s is a absolute path.'%path_rel)
    
    path_abs = os.path.abspath(os.path.join(path_start, path_rel))
    #file_path_from_path_start = os.path.relpath(path_rel, start=path_start)
    
    path_from_main = os.path.relpath(path_abs, start=path_main)

    #print('path_abs: %s path_main: %s path_from_main: %s'%(path_abs, path_main, path_from_main))
    '''
    print(main_path)
    print(path_start)
    print('path_start_rel: %s'%path_start_rel)
    print(file_name)
    print('file_path: %s'%file_path)
    #print('file_path_from_path_start: %s'%file_path_from_path_start)
    print('file_path_from_main_path: %s'%file_path_from_main_path)
    print(TargetDir_module(file_path_from_main_path))
    '''
    #print('path_rel: %s path_start: %s path_main: %s'%(path_rel, path_start, path_main))
    return path_from_main

def LoadBinaryFilePickle(FilePath):
    import pickle
    with open(FilePath, 'rb') as fo:
        Obj = pickle.load(fo, encoding='bytes')
    return Obj

def File2MD5(FilePath):
    import hashlib
    MD5Calculator = hashlib.md5()
    assert utils_torch.ExistsFile(FilePath), FilePath
    with open(FilePath, 'rb') as f:
        bytes = f.read()
    MD5Calculator.update(bytes)
    MD5Str = MD5Calculator.hexdigest()
    return MD5Str

def FileList2MD5(FilePathList):
    MD5List = []
    for FilePath in FilePathList:
        MD5 = File2MD5(FilePath)
        MD5List.append(MD5)
    return MD5List

def ListFilesAndCalculateMD5(Dir):
    Files = utils_torch.ListAllFiles(Dir)
    Dict = {}
    for File in Files:
        MD5 = utils_torch.files.File2MD5(Dir + File)
        Dict[File] = MD5
    return Dict