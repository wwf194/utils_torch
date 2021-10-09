import torch
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

def ProcessMNIST(dataset_dir, augment=True, batch_size=64):    
    transform = transforms.Compose(
    [transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=dataset_dir, transform=transform, train=True, download=False)
    testset = torchvision.datasets.MNIST(root=dataset_dir, transform=transform, train=False, download=False)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader
