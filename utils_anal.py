import torch

import matplotlib as mpl
from matplotlib import pyplot as plt

def plot_weight_2d(weight, ax=None, title=None):
    # 
    if ax is None:
        ax, fig = plt.subplots()



    if title is not None:
        ax.set_title()
        
    return 

def get_data_stat():
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

def plot_weight():
    return

