from resnet import resnet as resnet_cifar
import torch
import torch.backends.cudnn as cudnn
from torchvision.models.resnet import resnet50
from typing import *


# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet110", "imagenet32_resnet110"]

def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).to(device)
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).to(device)
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).to(device)
    elif arch == "imagenet32_resnet110":
        model = resnet_cifar(depth=110, num_classes=1000).to(device)

    # Both layers work fine, We tried both, and they both
    # give very similar results
    # IF YOU USE ONE OF THESE FOR TRAINING, MAKE SURE
    # TO USE THE SAME WHEN CERTIFYING.
    normalize_layer = get_normalize_layer(dataset)
    # normalize_layer = get_input_center_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)


_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

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
