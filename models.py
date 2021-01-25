"""
models.py
Model architectures for generator and discriminator
"""

import numpy as np
import torchlayers as tl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.distributions.multivariate_normal import MultivariateNormal


class Discriminator(nn.Module):
    """Discriminator that takes in two inputs, image and label information
        nc: number of channels
        ns: label size used for auxiliary head
        width: width parameter for model to scale general width
        use_spectral_norm: whether to use spectral normalization
    """
    def __init__(self, nc=3, ns=5, width=1, use_spectral_norm=True):
        super().__init__()
        
        self.nc = nc
        self.ns = ns

        def _spectral_norm(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                m = spectral_norm(m)

        self.body = nn.Sequential(
                nn.Conv2d(nc, 32 * width, 4, 2, 1), nn.LeakyReLU(),
                nn.Conv2d(32 * width, 32 * width, 4, 2, 1), nn.LeakyReLU(),
                nn.Conv2d(32 * width, 64 * width, 4, 2, 1), nn.LeakyReLU(),
                nn.Conv2d(64 * width, 64 * width, 4, 2, 1), nn.LeakyReLU(),
                nn.Flatten(),
        )
        
        self.aux = nn.Sequential(
            nn.Linear(ns, 128 * width), nn.LeakyReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(1152 * width, 128 * width), nn.LeakyReLU(),
            nn.Linear(128 * width, 128 * width), nn.LeakyReLU(),
            nn.Linear(128 * width, 1, bias=False)
        )

        if use_spectral_norm:
            self.body.apply(_spectral_norm)
            self.aux.apply(_spectral_norm)
            self.head.apply(_spectral_norm)
        
        print("Building discriminator...")


    def forward(self, x, y):
        hx = self.body(x)
        hy = self.aux(y)
        o = self.head(torch.cat((hx, hy), dim=-1))
        return o
   

    def args_dict(self):
        model_args = {
                        'nc': self.nc,
                        'ns': self.ns,
                     }

        return model_args


class View(nn.Module):
    """Extra module that flattens, mimicks TensorFlow behavior"""
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)


class Generator(nn.Module):
    """
    Generator module
    nz: size of latent
    nc: num channels
    bn: whether to use batchnorm
    """
    def __init__(self, nz=100, nc=3, bn=True):
        super().__init__()
        self.nz = nz
        self.nc = nc
        self.bn = bn

        def linearblock(num_feat, in_feat=None):
            layers = [nn.Linear(in_feat, num_feat)]
            if self.bn:
                layers.append(nn.BatchNorm1d(num_feat))
            layers.append(nn.ReLU(inplace=True))
            return layers
        
        def deconvblock(num_feat, in_feat=None, kernel=4, stride=2, padding=1):
            if self.use_nn:
                layers = [nn.ConvTranspose2d(in_feat, num_feat, kernel_size=kernel, stride=stride, padding=padding)]
                if self.bn:
                    layers.append(nn.BatchNorm2d(num_feat))
            else:
                layers = [tl.ConvTranspose2d(num_feat, kernel_size=kernel, stride=stride, padding=padding)]
                if self.bn:
                    layers.append(tl.BatchNorm())
            layers.append(nn.LeakyReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *linearblock(128, in_feat=nz),
            *linearblock(4 * 4 * 64, in_feat=128),
            View(-1, 64, 4, 4),
            *deconvblock(64, in_feat=64),
            *deconvblock(32, in_feat=64),
            *deconvblock(32, in_feat=32),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),
            nn.Sigmoid(),
        )
        
        print("Building generator...")

    
    def forward(self, z):
        return self.model(z)
    
    
    def args_dict(self):
        model_args = {
                        'nz': self.nz,
                        'nc': self.nc,
                        'bn': self.bn,
                        'use_nn': self.use_nn,
                     }

        return model_args
