import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR,ReduceLROnPlateau

import torchvision
from torchvision.datasets import mnist
import torchvision.transforms as transforms

from mylib_param_config import *

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import time
import os

def train(net, epochs, trainloader, testloader, train_loss_list_0=[], train_acc_list_0=[], val_loss_list_0=[], val_acc_list_0=[], save_dir="undefined", save=True, evaluate_before_train=True, save_interval=20, evaluate=None, logger=None, mode_name="model"):
    train_loss_list = [0.0 for _ in range(epoch_num)]
    train_acc_list = [0.0 for _ in range(epoch_num)]
    val_loss_list = [0.0 for _ in range(epoch_num)]
    val_acc_list = [0.0 for _ in range(epoch_num)]

    if(evaluate is None):
        evaluate = evaluate_iter

    if(isinstance(epochs, list)):
        epoch_start=epochs[0]
        epoch_end=epochs[1]
    elif(isinstance(epochs, int)):
        epoch_start=0
        epoch_end=epochs
    else:
        print("invalid epochs type.")

    if evaluate_before_train:
        with torch.no_grad():
            val_loss, val_acc = evaluate(net, testloader, criterion, scheduler, augment, device)
            note2 = ' val_loss:%.4f val_acc:%.4f'%(val_loss, val_acc)
            val_loss_list[epoch]=val_loss
            val_acc_list[epoch]=val_acc
            if(save==True and epoch%save_interval==0):
                net_path=save_dir_stat+model_name+"_epoch_%d_0/"%(epoch_start)
                if not os.path.exists(net_path):
                    os.makedirs(net_path)
                #torch.save(net.state_dict(), net_path + "torch_save.pth")
                net.save(net_path)
    
    for epoch in range(epoch_start, epoch_end+1):
        note0 = 'epoch=%d'%(epoch)

        loss_total = 0.0
        labels_count=0
        correct_count=0
        count=0
        for i, data in enumerate(trainloader, 0):
            #print("\r","progress:%d/50000 "%(labels_count), end="", flush=True)
            count=count+1
            inputs, labels = data
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = list(map(lambda x:x.to(device), outputs))

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct_count += (torch.max(outputs[-1], 1)[1]==labels).sum().item()
            labels_count += labels.size(0)
            loss_total += loss.item()

        train_loss_list[epoch]=train_loss
        train_acc_list[epoch]=train_acc

        note1='train_loss:%.4f train_acc:%.4f'%(train_loss, train_acc)

        with torch.no_grad():
            val_loss, val_acc = evaluate(net,testloader,criterion,scheduler,augment, device)
            note2 = ' val_loss:%.4f val_acc:%.4f'%(val_loss, val_acc)
            val_loss_list[epoch]=val_loss
            val_acc_list[epoch]=val_acc
            if(save==True and epoch%save_interval==0):
                net_path=save_dir_stat+model_name+"_epoch_%d/"%(epoch)
                if not os.path.exists(net_path):
                    os.makedirs(net_path)
                #torch.save(net.state_dict(), net_path + "torch_save.pth")
                net.save(net_path)

        logger.write(note0+note1+note2)

    with torch.no_grad():
        val_loss, val_acc = evaluate(net,testloader,criterion,scheduler,augment, device)
        note2 = ' val_loss:%.4f val_acc:%.4f'%(val_loss, val_acc)
        val_loss_list[epoch]=val_loss
        val_acc_list[epoch]=val_acc
        if(save==True and epoch%save_interval==0):
            net_path=save_dir_stat+model_name+"_epoch_%d/"%(epoch_end)
            if not os.path.exists(net_path):
                os.makedirs(net_path)
            #torch.save(net.state_dict(), net_path + "torch_save.pth")
            net.save(net_path)

    train_loss_list_0 = train_acc_list_0 + train_loss_list
    train_acc_list_0 = train_acc_list_0 + train_acc_list
    val_loss_list_0 = val_loss_list_0 + val_loss_list
    val_acc_list_0 = val_acc_list_0 + val_acc_list

def test():
    print("MyLib test.")
def pytorch_info():
    if(torch.cuda.is_available()==True):
        print("Cuda is available")
    else:
        print("Cuda is unavailable")

    print("Torch version is "+torch.__version__)

def prepare_MNIST(dataset_dir=MNIST_dir, augment=True, batch_size=64):    
    transform = transforms.Compose(
    [transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(root=dataset_dir, transform=transform, train=True, download=False)
    testset = torchvision.datasets.MNIST(root=dataset_dir, transform=transform, train=False, download=False)

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

def prepare_CIFAR10(dataset_dir=CIFAR10_dir,  norm=True, augment=False, batch_size=64, download=False):
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
'''
def prepare_CIFAR10(dataset_dir=CIFAR10_dir, norm=True, augment=False, batch_size=64):
    if(augment==True):
        feature_map_width=24
    else:
        feature_map_width=32

    if(augment==True):
        transform_train = transforms.Compose([
            transforms.RandomCrop(24),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose([
            transforms.TenCrop(24),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(crop) for crop in crops]))
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

    trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, transform=transform_train, download=False)
    testset = torchvision.datasets.CIFAR10(root=dataset_dir,train=False, transform=transform_test, download=False)
    
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=0)

    return trainloader, testloader
'''
def evaluate(net, testloader, criterion, scheduler, augment, device):
    net.eval()
    count=0
    labels_count=0
    correct_count=0
    labels_count=0
    loss_total=0.0
    #torch.cuda.empty_cache()
    for data in testloader:
        #print("\r","progress:%d/%d "%(count,len(testloader)), end="", flush=True)
        count=count+1
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)
        if(augment==True):
            bs, ncrops, c, h, w = inputs.size()
            outputs = net(inputs.view(-1, c, h, w))
            outputs = outputs.view(bs, ncrops, -1).mean(1)
        else:
            outputs = net(inputs) 
            outputs = outputs.to(device)
        loss_total += criterion(outputs, labels).item()
        correct_count+=(torch.max(outputs, 1)[1]==labels).sum().item()
        labels_count+=labels.size(0)
    #print("\n")
    val_loss=loss_total/count
    val_acc=correct_count/labels_count
    net.train()
    return val_loss, val_acc

def evaluate_iter(net, testloader, criterion, scheduler, augment, device):
    net.eval()
    count=0
    labels_count=0
    correct_count=0
    labels_count=0
    loss_total=0.0
    #torch.cuda.empty_cache()
    for data in testloader:
        #print("\r","progress:%d/%d "%(count,len(testloader)), end="", flush=True)
        count=count+1
        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)
        if(augment==True):
            bs, ncrops, c, h, w = inputs.size()
            outputs = net(inputs.view(-1, c, h, w))
            outputs = outputs.view(bs, ncrops, -1).mean(1)
        else:
            outputs, act = net(inputs)
            #outputs = list(map(lambda x:x.to(device), outputs))  
        loss_total += net.get_loss(inputs, labels).item()
        correct_count+=(torch.max(outputs[-1], 1)[1]==labels).sum().item()
        labels_count+=labels.size(0)
    #print("\n")
    val_loss=loss_total/count
    val_acc=correct_count/labels_count
    net.train()
    return val_loss, val_acc

'''
class model(nn.Module):
    def __init__(self):
        super(net_model, self).__init__()
        self.w1 = torch.nn.Parameter(1e-3*torch.rand(784, hidden_layer_size, device=device))
        self.b1 = torch.nn.Parameter(torch.zeros(hidden_layer_size, device=device))
        self.r1 = torch.nn.Parameter(1e-3*torch.rand(hidden_layer_size, hidden_layer_size,device=device))

        self.w2 = torch.nn.Parameter(1e-3*torch.rand(512, 10, device=device))
        self.b2 = torch.nn.Parameter(torch.zeros(10,device=device))

        
        self.mlp = torch.nn.Linear(96, 96, bias=True)

        self.relu = torch.nn.ReLU()

    def forward(self, x): # inputs : [batchsize, input_dim, time_windows]
        #change shape
        x = inputs.view(-1, 28 * 28)
        batch_size=x.size(0)
        input_dim=x.size(1)
        h = torch.zeros(batch_size, hidden_layer_size, device=device)
        for step in range(iter_time):
            h=h.mm(self.r1)            
            h=x+h+b1
            h=self.relu(h)
        
        x=h.mm(w2)+b2
        return x
'''

def prep_net(batch_size=64, load=False, model_name=''):
    if not load:
        model=model()
    else:
        net=torch.load(model_name)
        print('loading model:'+model_name)

    batch_size=64
    net.train()
    return net

def print_training_curve(train_loss_list, train_acc_list, val_loss_list, val_acc_list):
    plt.subplot(2, 1, 1)
    plt.plot(x, train_loss_list, '-', label='train', color='r')
    plt.plot(x, val_loss_list, '-', label='test', color='b')
    plt.title('Loss')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    
    plt.subplot(2, 1, 2)
    plt.plot(x, train_acc_list, '-', label='train', color='r')
    plt.plot(x, val_acc_list, '-', label='test', color='b')
    plt.title('Accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(loc='best')

    plt.tight_layout(pad=0.4, w_pad=2.0, h_pad=2.0)
    
    plt.savefig("train_statistics_"+model_name+".jpg")


    print(net.parameters())
    for parameters in net.parameters():
        print(parameters)

    for name,parameters in net.named_parameters():
        print(name,':',parameters.size())

    plt.hist(r1, normed=True, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("weights")
    plt.ylabel("frequency")
    plt.title("r1 Distribution")
    plt.savefig("r1_hist.jpg")




