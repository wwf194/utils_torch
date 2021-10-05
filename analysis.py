import torch

import matplotlib as mpl
from matplotlib import pyplot as plt

import utils_torch

def plot_weight_2d(weight, ax=None, title=None):
    # 
    if ax is None:
        ax, fig = plt.subplots()



    if title is not None:
        ax.set_title()
        
    return 

def Getdata_stat():
    print(torch.__version__)
    if torch.cuda.is_available():
        print('cuda is available')
    else:
        print('cuda is not available')
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    #MNIST
    #batch_size, batch_size_test, net, trainset, testset, trainloader, testloader, criterion, optimizer = prepare_MNIST(device, load=True)
    
    #CIFAR-10
    batch_size, net, trainset, testset, trainloader, testloader, criterion, optimizer = prepare_CIFAR_10(device, load=False)

    count=0
    mean=0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)
        # zeros the paramster gradients
        count=count + 1
        mean += torch.mean(inputs, (3,2,1,0))

    print('mean:%.9f'%(mean/(count*1.0)))

def AnalyzeTimeVaryingActivitiesEpochBatch(Logs, PlotIndex=0, EpochIndex=None, BatchIndex=None, SaveDir=None):
    if SaveDir is None:
        SaveDir = utils_torch.GetSaveDir()
    PlotIndex = 0
    for name, activity in Logs.items():
        EpochIndex = activity[0]
        BatchIndex = activity[1]
        activity = activity[2]
        _name = "%s-Epoch%d-Batch%d-No%d"%(name, EpochIndex, BatchIndex, PlotIndex)
        utils_torch.analysis.AnalyzeTimeVaryingActivity(
            activity,
            PlotIndex=PlotIndex,
            Name=_name, 
            SavePath=utils_torch.GetSaveDir() + "%s/%s.svg"%(name, _name),
        )

def AnalyzeTimeVaryingActivity(activity, PlotIndex, Name=None, SavePath=None):
    utils_torch.plot.PlotActivityAndDistributionAlongTime(
        axes=None,
        activity=activity,
        activityPlot=activity[PlotIndex],
        Title=Name,
        Save=True,
        SavePath=SavePath,
    )

def AnalyzeWeightsEpochBatch(Logs, EpochIndex=None, BatchIndex=None, PlotIndex=0, SaveDir=None):
    if SaveDir is None:
        SaveDir = utils_torch.GetSaveDir()
    PlotIndex = 0

    for Name, Weights in Logs.items():
        if isinstance(Weights, list):
            EpochIndex = Weights[0]
            BatchIndex = Weights[1]
            weights = Weights[2]
        for name, weight in weights.items():
            _name = "%s-Epoch%d-Batch%d-No%d"%(name, EpochIndex, BatchIndex, PlotIndex)
            utils_torch.analysis.AnalyzeWeight(
                weight,
                Name=_name, 
                SavePath=utils_torch.GetSaveDir() + "%s/%s.svg"%(name, _name),
            )

def AnalyzeWeight(weight, Name, SavePath=None):
    utils_torch.plot.PlotWeightAndDistribution(
        axes=None,
        weight=weight,
        Name=Name,
        SavePath=SavePath,
    )
    return

def AnalyzeWeightChange():
    return