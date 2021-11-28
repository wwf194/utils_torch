def ProcessMNIST(dataset_dir, augment=True, batch_size=64):    
    transform = transforms.Compose(
    [transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=dataset_dir, transform=transform, train=True, download=False)
    testset = torchvision.datasets.MNIST(root=dataset_dir, transform=transform, train=False, download=False)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader