import sys
import numpy as np
import utils_torch
from utils_torch.attrs import *

DatasetConfigFile = utils_torch.files.GetFileDir(__file__) + "CIFAR10.jsonc"
DatasetConfig = utils_torch.json.JsonFile2PyObj(DatasetConfigFile)

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
    DataObj = utils_torch.PyObj({
        "Labels": Labels,
        "Images": Images,
        "FileNames": FileNames
    })
    ImageNum = Images.shape[0]
    SetAttrs(DataObj, "Images.Num", value=ImageNum)

class DataLoaderForEpochBatchTraining:
    def __init__(self, param):
        utils_torch.model.InitForNonModel(self, param)
        return
    def InitFromParam(self, IsLoad=False):
        utils_torch.model.InitFromParamForNonModel(self, IsLoad)
        return
    def ApplyTransformOnData(self, TransformParam="Auto"):
        param = self.param
        cache = self.cache
        if TransformParam in ["Auto"]:
            TransformParam = param.Data.Transform
        assert hasattr(cache, "Data")
        Images = GetAttrs(cache.Data.Images)
        for Transform in TransformParam.Methods:
            if Transform.Type in ["Norm2Mean0Std1"]:
                EnsureAttrs(Transform, "Axis", None)
                Images = utils_torch.math.Norm2Mean0Std1(Images, Axis=Transform.Axis)
            else:
                raise Exception(Transform.Type)
        SetAttrs(cache.Data, "Images", value=Images)
        cache.Images = Images
    def NotifyEpochIndex(self, EpochIndex):
        self.EpochIndex = EpochIndex
    def LoadData(self, Dir="Auto"):
        cache = self.cache
        if Dir in ["Auto", "auto"]:
            Dir = utils_torch.GetDatasetDir("CIFAR10")
        cache.Data = utils_torch.json.DataFile2PyObj(Dir)
        #cache.IndexMax = cache.Data.Images.Num
    def PrepareBatches(self, BatchParam):
        cache = self.cache
        cache.IndexCurrent = 0
        cache.BatchSize = BatchParam.Batch.Size
        cache.BatchNum = utils_torch.dataset.CalculateBatchNum(cache.BatchSize, cache.Data.Images.Num)
        cache.IndexMax = cache.Data.Images.Num
        return
    def ClearBatches(self):
        cache = self.cache
        RemoveAttrIfExists(cache, "BatchSize")
        RemoveAttrIfExists(cache, "BatchNum")
        RemoveAttrIfExists(cache, "IndexCurrent")
        RemoveAttrIfExists(cache, "IndexMax")
    def GetBatch(self):
        cache = self.cache
        assert cache.IndexCurrent <= cache.IndexMax
        IndexStart = cache.IndexCurrent
        IndexEnd = min(cache.IndexCurrent + cache.Index, cache.IndexMax)
        DataBatch = {
            "Input": utils_torch.NpArray2Tensor(cache.Images[IndexStart:IndexEnd, :, :, :]).to(self.GetTensorLocation()),
            "Output": utils_torch.NpArray2Tensor(cache.Labels[IndexStart:IndexEnd]).to(self.GetTensorLocation()),
        }
        cache.IndexCurrent = IndexEnd
        return DataBatch
    def GetBatchNum(self):
        return self.cache.BatchNum
utils_torch.model.SetMethodForNonModelClass(DataLoaderForEpochBatchTraining, HasTensor=True)