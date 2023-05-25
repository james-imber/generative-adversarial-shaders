# Copyright:    Copyright (c) Imagination Technologies Ltd 2023
# License:      MIT (refer to the accompanying LICENSE file)
# Author:       AI Research, Imagination Technologies Ltd
# Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import utils
from PIL import Image

class TrainImageDataset(Dataset):
    def __init__(self, data_path_render, data_path_real, img_size, target_dataset):
        super(TrainImageDataset, self).__init__()
        self.image_render = utils.get_file_names(data_path_render)
        self.image_real = utils.get_file_names(data_path_real)

        self.crop = transforms.RandomCrop(img_size)
        self.hflip = transforms.RandomHorizontalFlip(p=0.5)

        self.target_dataset = target_dataset

    def __getitem__(self, batch_index):
        # Load the images

        # random index is used for cases when datasets have different lengthsss
        render_index = np.random.rand()
        render = Image.open(self.image_render[int(render_index * len(self.image_real))])
        real = Image.open(self.image_real[batch_index])

        # Remove artifacts at the borders of cityscapes images.
        if self.target_dataset == "cityscapes":
            real = transforms.functional.center_crop(real, (900, 1900))

        # Apply augmentation functions
        render = self.crop(render)
        real = self.crop(real)

        render = self.hflip(render)
        real = self.hflip(real)

        # Convert the images to torch tensors
        render = utils.image_to_tensor(render, integer=False)
        real = utils.image_to_tensor(real, integer=False)

        return {'render':render, 'real':real}

    def __len__(self):
        return len(self.image_real)

