from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://github.com/p3i0t/SimCLR-CIFAR10/blob/master/README.md
def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def get_color_distortion():
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

# https://github.com/p3i0t/SimCLR-CIFAR10/blob/master/README.md
def nt_xent(x, t=0.5):  #x shape -> 256x128
    x = F.normalize(x, dim=1)
    x_scores =  (x @ x.t()).clamp(min=1e-7)
    x_scale = x_scores / t 
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1

    targets[1::2] -= 1
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    model.to(device)

    # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
    with torch.no_grad():
        # Loop through the data
        for x, y in loader:

            # Move data to device
            x = x.to(device=device)
            y = y.to(device=device)

            # Forward pass
            scores = model(x)
            predictions = torch.argmax(scores,dim=1)

            # Check how many we got correct
            num_correct += (predictions == y).sum()

            # Keep track of number of samples
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples
