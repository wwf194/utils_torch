import sys
import numpy as np
import torch
import utils_torch
from utils_torch.attrs import *

DatasetConfigFile = utils_torch.files.GetFileDir(__file__) + "cifar10.jsonc"
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

class DataManagerForEpochBatchTrain:
    def __init__(self, param, **kw):
        utils_torch.module.InitForNonModel(self, param, **kw)
        return
    def InitFromParam(self, IsLoad=False):
        utils_torch.module.InitFromParamForNonModel(self, IsLoad)
        cache = self.cache
        cache.Flows = utils_torch.EmptyPyObj()
        # self.CreateFlowRandom("DefaultTest", "Test")
        # self.CreateFlowRandom("DefaultTrain", "Train")
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
    #def CreateFlow(self, Name, BatchParam, Type="Train"):
    def HasFlow(self, Name):
        return hasattr(self.cache.Flows, Name)
    def GetFlow(self, Name):
        return getattr(self.cache.Flows, Name)
    def CreateFlow(self, BatchParam, Name, Type="Train", IsRandom=False):
        cache = self.cache
        if self.HasFlow(Name):
            utils_torch.AddWarning("Overwriting existing flow: %s"%Name)
        #self.ClearFlow(Type=Type)
        flow = SetAttr(cache.Flows, Name, utils_torch.EmptyPyObj())
        flow.IndexCurrent = 0
        flow.BatchSize = BatchParam.Batch.Size
        Data = getattr(cache.Data, Type)
        flow.BatchNumMax = utils_torch.dataset.CalculateBatchNum(flow.BatchSize, Data.Images.Num)
        flow.IndexMax = Data.Images.Num
        flow.Data = Data
        flow.Images = GetAttrs(flow.Data.Images)
        flow.Labels = GetAttrs(flow.Data.Labels)
        if hasattr(BatchParam, "Batch.Num"): # Limited Num of Batches
            flow.BatchNum = BatchParam.Batch.Num
        else: # All
            flow.BatchNum = flow.BatchNumMax
        flow.BatchIndex = -1
        if IsRandom:
            flow.IsRandom = True
            flow.RandomBatchOrder = utils_torch.RandomOrder(range(flow.BatchNum))
            flow.RandomBatchIndex = 0
        else:
            flow.IsRandom = False
        return flow
    def CreateFlowRandom(self, BatchParam, Name, Type):
        return self.CreateFlow(BatchParam, Name, Type, IsRandom=True)
    #def ClearFlow(self, Type="Train"):
    def ClearFlow(self, Name):
        cache = self.cache
        if hasattr(cache.Flows, Name):
            delattr(cache.Flows, Name)
        else:
            utils_torch.AddWarning("No such flow: %s"%Name)
    def GetBatch(self, Name):
        self.GetBatch(self, self.GetFlow(Name))
    def GetBatchFromFlow(self, flow):
        flow.BatchIndex += 1
        assert flow.BatchIndex < flow.BatchNum
        if flow.IsRandom:
            return self.GetBatchRandomFromFlow(flow)
        assert flow.IndexCurrent <= flow.IndexMax
        IndexStart = flow.IndexCurrent
        IndexEnd = min(IndexStart + flow.BatchSize, flow.IndexMax)
        DataBatch = self.GetBatchFromIndex(flow, IndexStart, IndexEnd)
        flow.IndexCurrent = IndexEnd
        return DataBatch
    def GetData(self, Type):
        return getattr(self.cache.Data, Type)
    def GetBatchFromIndex(self, Data, IndexStart, IndexEnd):
        DataBatch = {
            "Input": utils_torch.NpArray2Tensor(
                    Data.Images[IndexStart:IndexEnd]
                ).to(self.GetTensorLocation()),
            "Output": utils_torch.NpArray2Tensor(
                    Data.Labels[IndexStart:IndexEnd],
                    DataType=torch.long # CrossEntropyLoss requires label to be LongTensor.
                ).to(self.GetTensorLocation()),
        }
        return DataBatch
    def GetBatchRandom(self, BatchParam, Type, Seed=None):
        BatchSize = BatchParam.Batch.Size
        Data = self.GetData(self, Type)
        if Seed is not None:
            IndexStart = Seed % (Data.Images.Num - BatchSize + 1)
        else:
            IndexStart = utils_torch.RandomIntInRange(0, Data.Images.Num - BatchSize)
        IndexEnd = IndexStart + BatchSize
        assert IndexEnd < Data.Images.Num
        return self.GetBatchFromIndex(self, Data, IndexStart, IndexEnd)
    def GetBatchRandomFromFlow(self, Name):
        flow = self.GetFlow(Name)
        assert flow.IsRandom
        IndexStart = flow.RandomBatchOrder[flow.RandomBatchIndex] * flow.BatchSize
        IndexEnd = min(IndexStart + flow.BatchSize, flow.IndexMax)
        DataBatch = self.GetBatchFromIndex(flow.Images, IndexStart, IndexEnd)
        flow.RandomBatchIndex += 1
        if flow.RandomBatchIndex > flow.IndexMax:
            flow.RandomBatchIndex = 0
        return DataBatch
    def ResetFlow(self, Name):
        flow = self.GetFlow(Name)
        flow.IndexCurrent = 0
        flow.BatchIndex = -1
    def GetBatchNum(self, Name="Train"):
        cache = self.cache
        flow = getattr(cache.Flows, Name)
        return flow.BatchNum
utils_torch.module.SetMethodForNonModelClass(DataManagerForEpochBatchTrain, HasTensor=True)

def ProcessCIFAR10(dataset_dir,  norm=True, augment=False, batch_size=64, download=False):
    if(augment==True):
        feature_map_width=24
    else:
        feature_map_width=32
        
    trans_train=[]
    trans_test=[]

    if(augment==True):
        TenCrop=[
            transforms.TenCrop(24),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(crop) for crop in crops]))
            ]
        trans_train.append(TenCrop)
        trans_test.append(TenCrop)

    trans_train.append(transforms.ToTensor())
    trans_test.append(transforms.ToTensor())

    if(norm==True):
        trans_train.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        trans_test.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    '''
    transforms.RandomCrop(24),
    transforms.RandomHorizontalFlip(),
    
    if(augment==True):
        transform_train = transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose()
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    '''
    transform_test=transforms.Compose(trans_test)
    transform_train=transforms.Compose(trans_train)
    
    trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, transform=transform_train, download=download)
    testset = torchvision.datasets.CIFAR10(root=dataset_dir,train=False, transform=transform_test, download=download)
    
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return trainloader, testloader