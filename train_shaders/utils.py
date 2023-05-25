# Copyright:    Copyright (c) Imagination Technologies Ltd 2023
# License:      MIT (refer to the accompanying LICENSE file)
# Author:       AI Research, Imagination Technologies Ltd
# Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


import os
import torch
from torchvision.transforms import functional as F 

def get_file_names(path):

    if len(next(os.walk(path))[1]) > 0:
        image_file_names = []

        for root, dir, files in os.walk(path):
            for name in files:
                image_file_names.append(os.path.join(root, name))
    else:
        file_names = os.listdir(path)
        image_file_names = [os.path.join(path, x) for x in file_names]

    return image_file_names

def image_to_tensor(image, norm=False, half=False, integer=False):
    """
    Takes an OpenCV image and converts it to a torch tensor.
    """

    tensor = F.to_tensor(image)
    # Scale from [0, 1] to [-1, 1]
    if norm:
        tensor = tensor.mul(2.0).sub(1.0)
    # Convert from torch.float32 to torch.float16
    if half:
        tensor = tensor.half()
    if integer:
        tensor *= 255
        tensor = tensor.type(torch.ByteTensor)
    return tensor

