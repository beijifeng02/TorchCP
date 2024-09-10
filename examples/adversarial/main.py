import os
import torch
import pickle
import gc
from sklearn.model_selection import train_test_split

from datasets import build_dataset
from model.models import get_architecture
from cfgs import cfg
from utils import set_seed, Smooth_Adv_ImageNet, calculate_accuracy_smooth, get_dimension

# images per batch
GPU_CAPACITY = 512

sigma_smooth = cfg["ratio"]["value"] * cfg["epsilon"]["value"]
correction = float(cfg["epsilon"]["value"]) / float(sigma_smooth)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_seed(0)
calibloader, testloader = build_dataset(batch_size=GPU_CAPACITY)
checkpoint = torch.load('examples/adversarial/pretrained_model/checkpoint.pth.tar', map_location=device)
model = get_architecture(checkpoint["arch"], "cifar10")
model.to(device)
model.eval()

indices = torch.arange(cfg["n_test"]["value"])

# get dimension of data
examples = enumerate(testloader)
batch_idx, (x_test, y_test) = next(examples)
rows = x_test.size()[2]
cols = x_test.size()[3]
channels = x_test.size()[1]

testloader = Smooth_Adv_ImageNet(model, x_test, y_test, indices, cfg["n_smooth"]["value"], sigma_smooth,
                    cfg["N_steps"]["value"], cfg["epsilon"]["value"], device, GPU_CAPACITY=GPU_CAPACITY)
testloader_base = Smooth_Adv_ImageNet(model, x_test, y_test, indices, 1, sigma_smooth,
                    cfg["N_steps"]["value"], cfg["epsilon"]["value"], device, GPU_CAPACITY=GPU_CAPACITY)


