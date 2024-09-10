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
GPU_CAPACITY = 256

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

x_test_adv = Smooth_Adv_ImageNet(model, testloader, indices, cfg["n_smooth"]["value"], sigma_smooth,
                    cfg["N_steps"]["value"], cfg["epsilon"]["value"], device, GPU_CAPACITY=GPU_CAPACITY)
x_test_adv_base = Smooth_Adv_ImageNet(model, testloader, indices, 1, sigma_smooth,
                    cfg["N_steps"]["value"], cfg["epsilon"]["value"], device, GPU_CAPACITY=GPU_CAPACITY)
os.makedirs(cfg["directory"]["value"])
with open(cfg["directory"]["value"] + "/data.pickle", 'wb') as f:
    pickle.dump([x_test_adv, x_test_adv_base], f)

# create the noises for the base classifiers only to check its accuracy
noises_base = torch.empty_like(x_test)
for k in range(cfg["n_test"]["value"]):
    torch.manual_seed(k)
    noises_base[k:(k + 1)] = torch.randn(
        (1, channels, rows, cols)) * sigma_smooth

# Calculate accuracy of classifier on clean test points
clean_acc, _, _ = calculate_accuracy_smooth(model, x_test, y_test, noises_base, cfg["num_of_classes"]["value"], k=1,
                                            device=device, GPU_CAPACITY=GPU_CAPACITY)
print("True Model accuracy :" + str(clean_acc * 100) + "%")

# Calculate accuracy of classifier on adversarial test points
adv_acc, _, _ = calculate_accuracy_smooth(model, x_test_adv_base, y_test, noises_base, cfg["num_of_classes"]["value"],
                                          k=1, device=device, GPU_CAPACITY=GPU_CAPACITY)
print("True Model accuracy on adversarial examples :" + str(adv_acc * 100) + "%")
del noises_base
gc.collect()

idx1, idx2 = train_test_split(indices, test_size=0.5)

