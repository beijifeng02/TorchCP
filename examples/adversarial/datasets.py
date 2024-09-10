import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def build_dataset(data_dir="/mnt/sharedata/ssd/common/datasets/", batch_size=512, num_workers=8, train=False):
    trainset = datasets.CIFAR10(root=data_dir, train=True, transform=transforms.ToTensor())
    testset = datasets.CIFAR10(root=data_dir, train=False, transform=transforms.ToTensor())

    dataset_length = len(testset)
    calb_length = dataset_length // 2
    calibset, testset = torch.utils.data.random_split(testset, [calb_length, dataset_length - calb_length])
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, num_workers=num_workers)
    calibloader = torch.utils.data.DataLoader(dataset=calibset, batch_size=calb_length, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=dataset_length-calb_length,
                                             num_workers=num_workers)

    if train:
        return trainloader, calibloader, testloader
    else:
        return calibloader, testloader
