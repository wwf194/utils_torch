import sys
import numpy as np
from attrs import EnsureAttrs
import utils_torch

utils_torch.GetFileDir()

DatasetConfig = utils_torch.PyObj({
    "Original":{
        "Files.MD5":{ # FileName: MD5Code
            "data_batch_1": "c99cafc152244af753f735de768cd75f", 
            "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
            "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
            "data_batch_4": "634d18415352ddfa80567beed471001a",
            "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
            "test_batch":   "40351d587109b95175f43aff81a1287e",
        },
        "Files.Train":[
            "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5",
        ],
        "Files.Test":[
            "test_batch",
        ],
        "Files.Num": 5
    }
})

def LoadOriginalFiles(Dir):
    Files = utils_torch.files.ListFiles(Dir)
    FileMD5s = DatasetConfig.Original.Files.MD5.ToDict()
    FileNames = DatasetConfig.Original.Files.Train + DatasetConfig.Original.Files.Test
    Dict = {}
    for File in Files:
        if File in FileNames:
            assert utils_torch.File2MD5(Dir + File) == FileMD5s[File]
            DataDict = utils_torch.files.LoadBinaryFilePickle(Dir + File)
            keys, values = utils_torch.Unzip(DataDict.items()) # items() cause logic error if altering dict in items() for-loop.
            for key, value in zip(keys, values):
                if isinstance(key, bytes):
                    DataDict[utils_torch.Bytes2Str(key)] = value # Keys in original dict are bytes. Turn them to string for convenience.
                    DataDict.pop(key)
            Dict[File] = DataDict
    assert len(Dict) != DatasetConfig.Original.Files.Num
    return Dict

def OriginalFiles2DataFile(LoadDir, SaveDir):
    OriginalDict = LoadOriginalFiles(LoadDir)
    DataDict = utils_torch.EmptyPyObj()
    DataDict.Train = ProcessOriginalDataDict(OriginalDict, FileNameList=DatasetConfig.Original.Files.Train)
    DataDict.Test = ProcessOriginalDataDict(OriginalDict, FileNameList=DatasetConfig.Original.Files.Test)
    utils_torch.json.PyObj2DataFile(DataDict, SaveDir)

def ProcessOriginalDataDict(Dict, FileNameList):
    Labels = []
    Images = []
    FileNames = []
    for File in FileNameList:
        Data = Dict[File]
        # Keys: batch_label, labels, data, filenames
        # Pixel values are integers with range [0, 255], so using datatype np.int8.
        # Saving as np.float32 will take ~ x10 disk memory as original files.
        _Images = utils_torch.ToNpArray(Data["data"], DataType=np.int8)
        _Images = _Images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        Images.append(_Images)
        _Labels = utils_torch.ToNpArray(Data["labels"], DataType=np.int8)
        Labels.append(_Labels)
        FileNames += Data["filenames"]
    Labels = np.concatenate(Labels, axis=0)
    Images = np.concatenate(Images, axis=0)
    return utils_torch.PyObj({
        "Labels": Labels,
        "Images": Images,
        "FileNames": FileNames
    })

class DataLoaderForEpochBatchTraining:
    def __init__(self):
        return
    def InitFromParam(self):
        return
    def ApplyTransformOnData(self, TransformParam="Auto"):
        param = self.param
        cache = self.cache
        if TransformParam in ["Auto"]:
            TransformParam = param.Data.Transform
        assert hasattr(cache, "Data")
        Images = cache.Data.Images
        for Transform in TransformParam.Methods:
            if Transform.Type in ["Norm2Mean0Std1"]:
                EnsureAttrs(Transform, "Axis", None)
                Images = utils_torch.math.Norm2Mean0Std1(Images, Axis=Transform.Axis)
            else:
                raise Exception(Transform.Type)
        cache.Data.Images = Images
        cache.Images = Images
    def NotifyEpochIndex(self, EpochIndex):
        self.EpochIndex = EpochIndex
    def LoadData(self, Dir="Auto"):
        if Dir in ["Auto", "auto"]:
            Dir = utils_torch.GetDatasetDir("CIFAR10")
        self.cache.Data = utils_torch.json.DataFile2PyObj(Dir)
    def PrepareBatches(self, BatchParam):
        cache = self.cache
        cache.IndexStart = 0
        cache.BatchSize = BatchParam.Batch.Size
        return
    def GetBatch(self, ):
        
        return