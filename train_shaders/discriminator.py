# Copyright:    Copyright (c) Imagination Technologies Ltd 2023
# License:      MIT (refer to the accompanying LICENSE file)
# Author:       AI Research, Imagination Technologies Ltd
# Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


import torch 
from torch import nn
from torch.nn import functional as F
import math
import utils
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.utils import save_image
from torch.nn.utils import spectral_norm

class PatchGAN(nn.Module):
    """
    Define a PatchGAN discriminator (https://arxiv.org/abs/1611.07004v3)

    Args:
        - in_channels[int]: number of channels in input batch
        - ndf[int]: number of filters to be used in the first convolutional
          layer and consequently in following layers.
        - n_layers[int]: number of layers to be used in the feedforward path.
    """
    def __init__(self, in_channels=3, ndf=64, n_layers=3, spectral=False):
        super(PatchGAN, self).__init__()
        
        if spectral:
            sequence = [
                spectral_norm(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding='valid')), 
                nn.LeakyReLU(0.2, True)
            ]
        else:
            sequence = [
                nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding='valid'), 
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            if spectral:
                sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding='valid'))] 
            else:
                sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding='valid')]

            sequence += [
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        if spectral:
            sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding='valid'))]
        else:
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding='valid')]

        sequence += [
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        if spectral:
            sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding='valid'))]  # output 1 channel prediction map
        else:
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding='valid')]

        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        return self.model(x)

class PatchGANWithNoise(nn.Module):
    """
    PatchGAN implementation with 3 layers modified to introduce noise at the 
    output of each discriminator's layer.

    Args:
        - in_channels[int]: number of channels in input batch
        - ndf[int]: number of filters to be used in the first convolutional
          layer and consequently in following layers.
        - n_layers[int]: number of layers to be used in the feedforward path.
    """
    def __init__(self, in_channels=3, normalisation='InstanceNorm'):
        super(PatchGANWithNoise, self).__init__()

        if normalisation == 'InstanceNorm':
            norm_layer = nn.InstanceNorm2d
        elif normalisation == 'BatchNorm':
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = nn.Identity

        ndf = 64

        self.conv_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding='valid')),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding='valid')),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding='valid')),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_4 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=1, padding='valid')),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_5 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding='valid')),
        )
    
    def forward(self, x):
        x_1 = self.conv_1(x)
        x_1 = x_1 + torch.normal(0, 0.1, size=x_1.shape, device=x_1.device)
        x_2 = self.conv_2(x_1)
        x_2 = x_2 + torch.normal(0, 0.1, size=x_2.shape, device=x_2.device)
        x_3 = self.conv_3(x_2)
        x_3 = x_3 + torch.normal(0, 0.1, size=x_3.shape, device=x_3.device)
        x_4 = self.conv_4(x_3)
        x_4 = x_4 + torch.normal(0, 0.1, size=x_4.shape, device=x_4.device)
        y = self.conv_5(x_4)
        return y
