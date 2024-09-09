import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def build_dataset(data_dir="/mnt/sharedata/ssd/common/datasets/", batch_size=512, num_workers=8):
    trainset = datasets.CIFAR10(root=data_dir, train=True, transform=transforms.ToTensor())
    testset = datasets.CIFAR10(root=data_dir, train=False, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, num_workers=num_workers)
    return trainloader, testloader
