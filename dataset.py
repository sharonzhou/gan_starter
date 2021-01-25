"""
dataset.py
Define simple dataset class for dsprites
"""

import time
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

from disentanglement_lib.data.ground_truth import dsprites


class Dataset(data.Dataset):
    """Dataset class for dsprites"""
    def __init__(self, dataset_name, batch_size, random_seed=42):
        self.dataset_name = dataset_name
        self.random_state = np.random.RandomState(random_seed)
        
        self.dataset = dsprites.DSprites(list(range(1, 6)))
        self.transform = self._set_transforms()


    def _set_transforms(self, use_normalize=False):
        """Decide transformations to data to be applied"""
        transforms_list = []

        # Normalize to the mean and standard deviation all pretrained
        # torchvision models expect
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # 1) transforms PIL image in range [0,255] to [0,1],
        # 2) tranposes [H, W, C] to [C, H, W]
        if use_normalize:
            transforms_list += [transforms.ToTensor(), normalize]
        else:
            transforms_list += [transforms.ToTensor()]
        transform = transforms.Compose([t for t in transforms_list if t])
        return transform

    
    def __len__(self):
        """Required: specify dataset length for dataloader"""
        return len(self.dataset.images)


    def __getitem__(self, index):
        """Required: specify what each iteration in dataloader yields"""
        return self.dataset.sample()
        
   
