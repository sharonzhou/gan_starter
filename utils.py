"""
utils.py
Helper functions used for training and inference
"""

import os
import sys
import time
import numpy as np
import torchlayers as tl
from tqdm import tqdm
from torchsummary import summary

from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils

from models import Generator, Discriminator


def sample_noise(batch_size, device):
    """Sample noise vectors for batch on GPU/CPU"""
    return torch.randn(batch_size, device=device)


def visualize(images, save_path):
    """Visualize a grid of outputs and save to path"""
    gridded_images = vutils.make_grid(images, padding=2, normalize=True)
    vutils.save_image(gridded_images, save_path)
    print(f'Saved images to {save_path}')


def load_models(args, model_types=['generator', 'discriminator']):
    """Loop over different model types"""
    models = []
    for model_type in model_types:
        model = load_model(args, model_type)
        models.append(model)
    return models


def load_model(args):
    """Load each model and put on dataparallel to use on multiple gpus"""
    if model_type == 'generator':
        generator = Generator().to(args.device)
        generator = nn.DataParallel(generator, args.gpu_ids)
        return generator

    elif model_type == 'discriminator':
        discriminator = Discriminator().to(args.device)
        discriminator = nn.DataParallel(discriminator, args.gpu_ids)
        return discriminator
    

def load_optimizer(args, models):
    """Load optimizers for each specified model's parameters"""
    if not isinstance(models, list):
        models = [models]
    optimizers = []
    for model in models:
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        optimizers.append(opt)
    return optimizers if len(optimizers) > 1 else optimizers[0]
