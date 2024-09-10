import os
import random
import numpy as np
import torch
from tqdm import tqdm
import gc
from torch.nn.functional import softmax
from scipy.stats import rankdata
from scipy.stats.mstats import mquantiles
import warnings
import math

from attacks import PGD_L2, DDN


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def Smooth_Adv_ImageNet(model, dataloader, indices, n_smooth, sigma_smooth, N_steps=20, max_norm=0.125, device='cpu',
                        GPU_CAPACITY=1024, method='PGD'):
    # create attack model
    if method == 'PGD':
        attacker = PGD_L2(steps=N_steps, device=device, max_norm=max_norm)
    elif method == "DDN":
        attacker = DDN(steps=N_steps, device=device, max_norm=max_norm)

    # create container for the adversarial examples
    x_adv = torch.zeros_like(x)

    # get number of data points
    n = x.size()[0]

    # get dimension of data
    rows = x.size()[2]
    cols = x.size()[3]
    channels = x.size()[1]

    # number of permutations to estimate mean
    num_of_noise_vecs = n_smooth

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // num_of_noise_vecs

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # start generating examples for each batch
    print("Generating Adverserial Examples:")

    image_index = -1
    for j in tqdm(range(num_of_batches)):
        # GPUtil.showUtilization()
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]
        curr_batch_size = inputs.size()[0]

        # duplicate batch according to the number of added noises and send to device
        # the first num_of_noise_vecs samples will be duplicates of x[0] and etc.
        tmp = torch.zeros((len(labels) * num_of_noise_vecs, *inputs.shape[1:]))
        x_tmp = inputs.repeat((1, num_of_noise_vecs, 1, 1)).view(tmp.shape).to(device)

        # send labels to device
        y_tmp = labels.to(device).long()

        # generate random Gaussian noise for the duplicated batch
        noise = torch.empty((curr_batch_size * n_smooth, channels, rows, cols))
        # get relevant noises for this batch
        for k in range(curr_batch_size):
            image_index = image_index + 1
            torch.manual_seed(indices[image_index])
            noise[(k * n_smooth):((k + 1) * n_smooth)] = torch.randn(
                (n_smooth, channels, rows, cols)) * sigma_smooth

        # noise = noises[(j * (batch_size * num_of_noise_vecs)):((j + 1) * (batch_size * num_of_noise_vecs))].to(device)
        # noise = torch.randn_like(x_tmp, device=device) * sigma_adv

        noise = noise.to(device)
        # generate adversarial examples for the batch
        x_adv_batch = attacker.attack(model, x_tmp, y_tmp,
                                      noise=noise, num_noise_vectors=num_of_noise_vecs,
                                      no_grad=False,
                                      )

        # take only the one example for each point
        x_adv_batch = x_adv_batch[::num_of_noise_vecs]

        # move back to CPU
        x_adv_batch = x_adv_batch.to(torch.device('cpu'))

        # put in the container
        x_adv[(j * batch_size):((j + 1) * batch_size)] = x_adv_batch.detach().clone()

        del noise, tmp, x_adv_batch
        gc.collect()

    # return adversarial examples
    return x_adv



def calculate_accuracy_smooth(model, x, y, noises, num_classes, k=1, device='cpu', GPU_CAPACITY=1024):
    # get size of the test set
    n = x.size()[0]

    # number of permutations to estimate mean
    n_smooth = noises.size()[0] // n

    # create container for the outputs
    smoothed_predictions = torch.zeros((n, num_classes))

    # calculate maximum batch size according to gpu capacity
    batch_size = GPU_CAPACITY // n_smooth

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # get predictions over all batches
    for j in range(num_of_batches):
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]

        # duplicate batch according to the number of added noises and send to device
        # the first n_smooth samples will be duplicates of x[0] and etc.
        tmp = torch.zeros((len(labels) * n_smooth, *inputs.shape[1:]))
        x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

        # generate random Gaussian noise for the duplicated batch
        noise = noises[(j * (batch_size * n_smooth)):((j + 1) * (batch_size * n_smooth))].to(device)

        # add noise to points
        noisy_points = x_tmp + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1)

        # get smoothed prediction for each point
        for m in range(len(labels)):
            smoothed_predictions[(j * batch_size) + m, :] = torch.mean(
                noisy_outputs[(m * n_smooth):((m + 1) * n_smooth)], dim=0)

    # transform results to numpy array
    smoothed_predictions = smoothed_predictions.numpy()

    # get label ranks to calculate top k accuracy
    label_ranks = np.array([rankdata(-smoothed_predictions[i, :], method='ordinal')[y[i]] - 1 for i in range(n)])

    # get probabilities of correct labels
    label_probs = np.array([smoothed_predictions[i, y[i]] for i in range(n)])

    # calculate accuracy
    top_k_accuracy = np.sum(label_ranks <= (k - 1)) / float(n)

    # calculate average inverse probability score
    score = np.mean(1 - label_probs)

    # calculate the 90 qunatiule
    quantile = mquantiles(1-label_probs, prob=0.9)
    return top_k_accuracy, score, quantile


def get_device(model):
    """
    Get the device of Torch model.

    :param model: a Pytorch model. If None, it uses GPU when the cuda is available, otherwise it uses CPU。

    :return: the device in use
    """
    if model is None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            device = torch.device(f"cuda:{cuda_idx}")
    else:
        device = next(model.parameters()).device
    return device


def calculate_conformal_value(scores, alpha, default_q_hat=torch.inf):
    """
    Calculate the 1-alpha quantile of scores.

    :param scores: non-conformity scores.
    :param alpha: a significance level.

    :return: the threshold which is use to construct prediction sets.
    """
    if default_q_hat == "max":
        default_q_hat = torch.max(scores)
    if alpha >= 1 or alpha <= 0:
        raise ValueError("Significance level 'alpha' must be in [0,1].")
    if len(scores) == 0:
        warnings.warn(
            f"The number of scores is 0, which is a invalid scores. To avoid program crash, the threshold is set as {default_q_hat}.")
        return default_q_hat
    N = scores.shape[0]
    qunatile_value = math.ceil(N + 1) * (1 - alpha) / N
    if qunatile_value > 1:
        warnings.warn(
            f"The value of quantile exceeds 1. It should be a value in [0,1]. To avoid program crash, the threshold is set as {default_q_hat}.")
        return default_q_hat

    return torch.quantile(scores, qunatile_value, dim=0).to(scores.device)


def get_dimension(dataloader):
    examples = enumerate(dataloader)
    batch_idx, (x_test, y_test) = next(examples)
    rows = x_test.size()[2]
    cols = x_test.size()[3]
    channels = x_test.size()[1]
    return rows, cols, channels
