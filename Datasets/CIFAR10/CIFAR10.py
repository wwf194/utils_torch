import sys
import numpy as np
import torch
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
        # Pixel values are integers with range [0, 255], so using datatype np.uint8.
        # Saving as np.float32 will take ~ x10 disk memory as original files.
        _Images = utils_torch.ToNpArray(Data["data"], DataType=np.uint8)
        _Images = _Images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        Images.append(_Images)
        _Labels = utils_torch.ToNpArray(Data["labels"], DataType=np.uint8)
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
    return DataObj

class DataLoaderForEpochBatchTraining:
    def __init__(self, param, **kw):
        utils_torch.model.InitForNonModel(self, param, **kw)
        return
    def InitFromParam(self, IsLoad=False):
        utils_torch.model.InitFromParamForNonModel(self, IsLoad)
        return
    def ApplyTransformOnData(self, TransformParam="Auto", Type=["Train", "Test"]):
        utils_torch.AddLog("Applying transformation on dataset images...")
        param = self.param
        cache = self.cache
        if TransformParam in ["Auto"]:
            TransformParam = param.Data.Transform
        assert hasattr(cache, "Data")
        for _Type in Type:
            Data = getattr(cache.Data, _Type)
            Images = GetAttrs(Data.Images)
            for Transform in TransformParam.Methods:
                if Transform.Type in ["ToGivenDataType"]:
                    Images = utils_torch.ToGivenDataTypeNp(Images, DataType=Transform.DataType)
                elif Transform.Type in ["Color2Gray", "ColorImage2GrayImage"]:
                    Images = utils_torch.plot.ColorImage2GrayImage(Images, ColorAxis=3)
                elif Transform.Type in ["Norm2Mean0Std1"]:
                    EnsureAttrs(Transform, "axis", None)
                    Images = utils_torch.math.Norm2Mean0Std1Np(Images, axis=tuple(GetAttrs(Transform.axis)))
                elif Transform.Type in ["Flatten"]:
                    # Plot example images before Flatten, which is usually the last step.
                    utils_torch.plot.PlotExampleImage(Images, SaveDir=utils_torch.GetMainSaveDir() + "Dataset/", SaveName="CIFAR10-%s"%_Type)
                    Shape = Images.shape
                    Images = Images.reshape(Shape[0], -1)
                else:
                    raise Exception(Transform.Type)
            SetAttrs(Data, "Images", value=Images)        
        utils_torch.AddLog("Applied transformation on dataset images.")
    def Labels2ClassNames(self, Labels):
        ClassNames = []
        for Label in Labels:
            ClassNames.append()
    def Label2ClassName(self, Label):
        return
    def NotifyEpochIndex(self, EpochIndex):
        self.EpochIndex = EpochIndex
    def LoadData(self, Dir="Auto"):
        cache = self.cache
        if Dir in ["Auto", "auto"]:
            Dir = utils_torch.GetDatasetDir("CIFAR10")
        DataFile = Dir + "CIFAR10-Data"
        cache.Data = utils_torch.json.DataFile2PyObj(DataFile)
        return
    def EstimateBatchNum(self, BatchSize, Type="Train"):
        cache = self.cache
        Data = getattr(cache.Data, Type)
        return utils_torch.dataset.CalculateBatchNum(BatchSize, Data.Images.Num)
    def PrepareBatches(self, BatchParam, Type="Train"):
        cache = self.cache
        self.ClearBatches()
        cache.IndexCurrent = 0
        cache.BatchSize = BatchParam.Batch.Size
        Data = getattr(cache.Data, Type)
        cache.BatchNum = utils_torch.dataset.CalculateBatchNum(cache.BatchSize, Data.Images.Num)
        cache.IndexMax = Data.Images.Num
        cache.DataForBatches = Data
        cache.ImagesForBatches = GetAttrs(cache.DataForBatches.Images)
        cache.LabelsForBatches = GetAttrs(cache.DataForBatches.Labels)
        utils_torch.RemoveAttrIfExists(cache, "RandomBatchOrder")
        utils_torch.RemoveAttrIfExists(cache, "RandomBatchIndex")
        return
    def ClearBatches(self):
        cache = self.cache
        RemoveAttrIfExists(cache, "BatchSize")
        RemoveAttrIfExists(cache, "BatchNum")
        RemoveAttrIfExists(cache, "IndexCurrent")
        RemoveAttrIfExists(cache, "IndexMax")
    def GetBatch(self):
        cache = self.cache
        DataForBatches = cache.DataForBatches
        assert cache.IndexCurrent <= cache.IndexMax
        IndexStart = cache.IndexCurrent
        IndexEnd = min(IndexStart + cache.BatchSize, cache.IndexMax)
        DataBatch = {
            "Input": utils_torch.NpArray2Tensor(
                    cache.ImagesForBatches[IndexStart:IndexEnd]
                ).to(self.GetTensorLocation()),
            "Output": utils_torch.NpArray2Tensor(
                    cache.LabelsForBatches[IndexStart:IndexEnd],
                    DataType=torch.long # CrossEntropyLoss requires label to be LongTensor.
                ).to(self.GetTensorLocation()),
        }
        cache.IndexCurrent = IndexEnd
        return DataBatch
    def GetRandomBatch(self):
        cache = self.cache
        if not hasattr(cache, "RandomBatchOrder"):
            cache.RandomBatchOrder = utils_torch.RandomOrder(range(cache.IndexMax))
            cache.RandomBatchIndex = 0
        DataForBatches = cache.DataForBatches
        IndexStart = cache.RandomBatchOrder[cache.RandomBatchIndex] * cache.BatchSize
        IndexEnd = min(IndexStart + cache.BatchSize, cache.IndexMax)
        DataBatch = {
            "Input": utils_torch.NpArray2Tensor(
                    cache.ImagesForBatches[IndexStart:IndexEnd]
                ).to(self.GetTensorLocation()),
            "Output": utils_torch.NpArray2Tensor(
                    cache.LabelsForBatches[IndexStart:IndexEnd],
                    DataType=torch.long # CrossEntropyLoss requires label to be LongTensor.
                ).to(self.GetTensorLocation()),
        }
        cache.RandomBatchIndex += 1
        if cache.RandomBatchIndex > cache.IndexMax:
            cache.RandomBatchIndex = 0
        return DataBatch
    def GetBatchNum(self):
        return self.cache.BatchNum
utils_torch.model.SetMethodForNonModelClass(DataLoaderForEpochBatchTraining, HasTensor=True)