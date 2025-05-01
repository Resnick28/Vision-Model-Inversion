import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F

def weights_initialization_gen_custom(m):
    if isinstance(m, nn.ConvTranspose2d):
        # Xavier initialization for ConvTranspose2d layers
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # Constant initialization for BatchNorm2d layers
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # Orthogonal initialization for Linear (embedding) layers
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def add_linf_perturbation(images, perturbation_bound):
    # Generate random perturbation
    perturbation = torch.randn_like(images)
    perturbation = perturbation_bound * torch.sign(perturbation)
    perturbed_images = images + perturbation
    perturbed_images = torch.clamp(perturbed_images, min=1e-6)  # Ensure no zero or negative values
    return perturbed_images