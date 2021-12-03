import utils_torch
import utils_torch.dataset.cifar10 as cifar10
import utils_torch.dataset.mnist as mnist

ModuleList = [
    "CIFAR10",
    "MNIST",
    "MSCOCO",
]

def IsLegalModuleType(Type):
    return Type in ModuleList

def DataSetType2InputOutputOutput(Type):
    if Type in ["CIFAR10", "cifar10"]:
        return utils_torch.dataset.cifar10.DatasetConfig
    elif Type in ["MNIST", "mnist"]:
        return utils_torch.dataset.mnist.DatasetConfig
    else:
        raise Exception(Type)

ModuleList = set(ModuleList)

import torch
import utils_torch

def CalculateBatchNum(BatchSize, SampleNum):
    BatchNum = SampleNum // BatchSize
    if SampleNum % BatchSize > 0:
        BatchNum += 1
    return BatchNum

config = utils_torch.files.JsonFile2PyObj(
    utils_torch.files.GetFileDir(__file__) + "config.jsonc"
)

def GetDatasetPath(Name):
    if Name in ["CIFAR10", "cifar10"]:
        Name = "cifar10"
    elif Name in ["MNIST", "mnist"]:
        Name = "mnist"
    else:
        raise Exception(Name)

    assert hasattr(config, Name)
    return getattr(config, Name).path



