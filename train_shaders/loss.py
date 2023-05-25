# Copyright:    Copyright (c) Imagination Technologies Ltd 2023
# License:      MIT (refer to the accompanying LICENSE file)
# Author:       AI Research, Imagination Technologies Ltd
# Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


import torch 
from torch import nn
from torch.nn import functional as F

class LsganLoss(nn.Module):
    """
    Instantiate LSGAN Loss
    """
    def __init__(self, prediction_size):
        super(LsganLoss, self).__init__()
        # Define a tensor for real labels and fake labels
        # of same size as the discriminator output.
        self.register_buffer(
            'real_label', torch.ones(prediction_size, device=torch.device('cuda', 0)))
        self.register_buffer(
            'fake_label', torch.zeros(prediction_size, device=torch.device('cuda', 0)))

    def forward(self, pred, target):
        # if target is True, calculate MSE Loss against real 
        # labels, else fake labels.
        if target:
            loss = F.mse_loss(pred, self.real_label)
        else:
            loss = F.mse_loss(pred, self.fake_label)
        return loss

class GANLoss(nn.Module):
    """
    Define GAN Loss inspired by EnlightenGAN.
    """
    def __init__(self, prediction_size):
        super(GANLoss, self).__init__()
        self.lsgan_criterion = LsganLoss(prediction_size)

    @staticmethod
    def relativistic_loss(pred_x, pred_y):
        """
        Define a method to calculate relativistic discriminator
        loss.
        """
        return pred_x - torch.mean(pred_y)

    def forward(self, pred_x, pred_y):
        # Calculate relativitic discriminator loss for 
        # both fake and real predictions.
        l_r_x = self.relativistic_loss(pred_x, pred_y)
        l_r_y = self.relativistic_loss(pred_y, pred_x)
        # Calculate LSGAN loss based on the relativistic 
        # discriminator predictions.
        lsgan_x = self.lsgan_criterion(l_r_x, True)
        lsgan_y = self.lsgan_criterion(l_r_y, False)
        return (lsgan_x + lsgan_y)/2
