import utils_torch
import utils_torch.Datasets.CIFAR10.CIFAR10 as CIFAR10

ModuleList = [
    "CIFAR10",
    "MNIST",
    "MSCOCO",
]

def IsLegalModuleType(Type):
    return Type in ModuleList

def DataSetType2InputOutputOutput(Type):
    if Type in ["CIFAR10", "cifar10"]:
        return utils_torch.Datasets.CIFAR10.DatasetConfig
    else:
        raise Exception(Type)

ModuleList = set(ModuleList)