import utils_torch
import utils_torch.Datasets.CIFAR10.CIFAR10 as CIFAR10

def DataSetType2InputOutputOutput(Type):
    if Type in ["CIFAR10", "cifar10"]:
        return utils_torch.Datasets.CIFAR10.DatasetConfig
    else:
        raise Exception(Type)