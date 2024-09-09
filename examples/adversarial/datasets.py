import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from typing import *


_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def build_dataset(data_dir="/mnt/sharedata/ssd/common/datasets/", batch_size=512, num_workers=8):
    trainset = datasets.CIFAR10(root=data_dir, train=True, transform=transforms.ToTensor())
    testset = datasets.CIFAR10(root=data_dir, train=False, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, num_workers=num_workers)
    return trainloader, testloader


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds
