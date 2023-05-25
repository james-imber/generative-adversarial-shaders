# Copyright:    Copyright (c) Imagination Technologies Ltd 2023
# License:      MIT (refer to the accompanying LICENSE file)
# Author:       AI Research, Imagination Technologies Ltd
# Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


import torch 
from torch import nn
from torch.nn import functional as F

class ColourMapping(nn.Module):
    def __init__(self):
        super(ColourMapping, self).__init__()
        
        # Define colour transform to apply to image.
        transform = torch.unsqueeze(torch.eye(3, dtype=torch.float32), dim=0)
        shift_vector = torch.unsqueeze(torch.zeros((3, 1), dtype=torch.float32), dim=0)

        self.transform = nn.parameter.Parameter(transform)
        self.shift_vector = nn.parameter.Parameter(shift_vector)

    def forward(self, img):
        # Get image size
        img_size = img.size()

        img = torch.reshape(
            img, (img_size[0], img_size[1], img_size[2]*img_size[3]))
        
        img = torch.matmul(self.transform, img)

        # Add Colour bias
        img = torch.add(img, self.shift_vector)

        # Reshape image to B,C,H,W
        img = torch.reshape(
            img, (img_size[0], img_size[1], img_size[2], img_size[3]))

        return img

class GaussianBlur(nn.Module):
    """
    Define a trainable Gaussian Blur shader. The shader learns the variance of 
    the horizontal and vertical blurring kernels.

    Args:
        - kernel_size (int): size of the kernel
        - channels (int): number of input channels
        - sigma (float): variance value used to populate the kernel
        when instantiating the object.
    """
    def __init__(self, kernel_size, sigma=2.0, channels=3, blur_magnitude=False):
        super(GaussianBlur, self).__init__()

        # Define trainable sigmas and magintude for trainable blur.
        self.sigma_x = nn.parameter.Parameter(torch.empty(1))
        self.sigma_y = nn.parameter.Parameter(torch.empty(1))
        
        # Initialise sigmas randomly to help avoiding local minimums.
        nn.init.trunc_normal_(self.sigma_x, mean=1.0, std=0.5, a=0.0, b=2.0)
        nn.init.trunc_normal_(self.sigma_y, mean=1.0, std=0.5, a=0.0, b=2.0)

        if blur_magnitude:
            self.magnitude = nn.parameter.Parameter(torch.tensor([1.0]))
        else:
            self.register_buffer("magnitude", torch.tensor([1.0]))

        # Define number of channels and kernels size.
        self.C = channels
        self.K = kernel_size
        
    def gaussian_kernel_1d(self, sigma):
        """
        Utility function to build 1D Gaussian Kernels
        """
        half_kernel_size = self.K // 2
        # Get coordinates of kernel weights in the kernel
        x = torch.linspace(-half_kernel_size, half_kernel_size, steps=self.K).to(device=sigma.device)
        # Get weights magnitudes.
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()
        # Multiply kernel weights by trainable magnitude.
        kernel1d *= self.magnitude
            
        return kernel1d 
    
    def forward(self, input):
        # get weights for horizontal and vertical blur kernels
        x_vals = self.gaussian_kernel_1d(self.sigma_x)
        y_vals = self.gaussian_kernel_1d(self.sigma_y)
        h_kernel = torch.zeros((self.C, 1, 1, self.K)).to(device=input.device)
        v_kernel = torch.zeros((self.C, 1, self.K, 1)).to(device=input.device)
        h_kernel[:, 0, 0, :] = x_vals
        v_kernel[:, 0, :, 0] = y_vals

        # Apply horizontal blur
        y = F.conv2d(input, h_kernel, padding='same', groups=input.shape[-3])
        # Apply vertical blur
        y = F.conv2d(y, v_kernel, padding='same', groups=input.shape[-3])
        return y

class BloomMultiRes(nn.Module):
    """
    Multi-resolution Bloom. For each resolution, the module learns a sigmoid soft threshold 
    to isolate the lights and a Gaussian Blur to control the amount of bloom applied to the
    frame. The different resolutions are then blended using a learnable tone mapping curve.

    Args:
        - random_mask (bool): If True, initialise the sigmoid soft mask parameters randomly. 
        - bloom_magnitude (bool): If True, learn a blur magnitude for each Gaussian Kernel.
        - learnable_luma (bool): If True, learn luma conversion parameters.
        - return_cmap (bool): If True, return a tuple where the first argument is the bloom input
          and the second argument is the bloom input.
    """
    def __init__(self, random_mask=False, blur_magnitude=False, 
                 learnable_luma=True, return_cmap=False):
        
        super(BloomMultiRes, self).__init__()

        # Define Lightmask block for each resolution scale
        self.mask_1 = LightMask(random=random_mask, learnable_luma=learnable_luma)
        self.mask_2 = LightMask(random=random_mask, learnable_luma=learnable_luma)
        self.mask_4 = LightMask(random=random_mask, learnable_luma=learnable_luma)
        self.mask_8 = LightMask(random=random_mask, learnable_luma=learnable_luma)

        # Define Blur for each resolution scale
        self.blur_1 = GaussianBlur(5, blur_magnitude=blur_magnitude)
        self.blur_2 = GaussianBlur(5, blur_magnitude=blur_magnitude)
        self.blur_4 = GaussianBlur(5, blur_magnitude=blur_magnitude)
        self.blur_8 = GaussianBlur(5, blur_magnitude=blur_magnitude)

        # Set fixed convolution for downscaling the image by 2 times.
        downconv = torch.ones((3, 1, 2, 2))*0.25
        self.register_buffer('downconv', downconv)

        # Set trainable exposure parameter for rendering final image.
        self.exposure = nn.parameter.Parameter(torch.tensor([1.2]).float())
        self.avg = nn.parameter.Parameter(torch.tensor([0.25]).float())

        self.saturation = nn.parameter.Parameter(torch.tensor([1.2]).float())

        self.return_cmap = return_cmap

    def forward(self, image):
        
        # Get original image shape
        img_shape = image.shape
        # Downsample the original image to Res/2, Res/4 and Res/8
        image_2 = F.conv2d(image, self.downconv, stride=2, groups=img_shape[-3])
        image_4 = F.conv2d(image_2, self.downconv, stride=2, groups=img_shape[-3])
        image_8 = F.conv2d(image_4, self.downconv, stride=2, groups=img_shape[-3])

        # Get a light mask for each resolution scale
        mask_1 = self.mask_1(image)
        mask_2 = self.mask_2(image_2)
        mask_4 = self.mask_4(image_4)
        mask_8 = self.mask_8(image_8)
        
        # Blur each light mask 
        mask_1 = self.blur_1(mask_1)
        mask_2 = self.blur_2(mask_2)
        mask_4 = self.blur_4(mask_4)
        mask_8 = self.blur_8(mask_8)

        # Scale up each light mask to input image resolution
        mask_2 = F.interpolate(
            mask_2, size=(img_shape[-2], img_shape[-1]), mode='bicubic')
        mask_4 = F.interpolate(
            mask_4, size=(img_shape[-2], img_shape[-1]), mode='bicubic')
        mask_8 = F.interpolate(
            mask_8, size=(img_shape[-2], img_shape[-1]), mode='bicubic')

        # Sum all the masks together
        mask = mask_1 + mask_2 + mask_4 + mask_8
        # Get a blend of input image and masks
        blend = image + mask

        result = torch.ones(blend.shape, dtype=torch.float32, device=blend.device)

        result -= torch.exp(torch.clamp(-F.relu(blend) * self.exposure, min=-10.0))
        sat_point = torch.exp(torch.clamp(self.exposure * self.saturation, min=-10.0))

        result *= (sat_point / (sat_point - 1))
        result = torch.clamp(result, min=1e-5, max=1.0)

        if self.return_cmap:
            return result, image

        return result

class LightMask(nn.Module):
    """
    Use a Sigmoid to isolate light sources in an image with 
    smooth thresholding.

    Args:
        - random (bool): if True, initialise smooth threshold parameters randomly.
        - learnable_luma (bloom): if True, learn luma conversion parameters.
    """
    def __init__(self, random=False, learnable_luma=True):
        super(LightMask, self).__init__()
        # Define trainable parameters for sigmoid, such that:
        #             1 
        # S(x) = -------------
        #        1 - e^-b*(Y-a)
        
        if random:
            self.a = nn.parameter.Parameter(torch.empty(1))
            self.b = nn.parameter.Parameter(torch.empty(1))
        
            nn.init.trunc_normal_(self.a, mean=0.8, std=2.0, a=0.0, b=2.0)
            nn.init.trunc_normal_(self.b, mean=50.0, std=2.0, a=25.0, b=75.0)
        else:
            self.a = nn.parameter.Parameter(torch.tensor([0.75]).float())
            self.b = nn.parameter.Parameter(torch.tensor([37.5]).float())

        if learnable_luma:self.Y_weight = nn.parameter.Parameter(torch.Tensor([[0.299], [0.587], [0.114]]))
        else: self.register_buffer("Y_weight", torch.Tensor([[0.299], [0.587], [0.114]]))

    def rgb2ycbcr(self, image):
        Y = torch.matmul(image.permute(0, 2, 3, 1), F.softmax(self.Y_weight, dim=0)).permute(0, 3, 1, 2)
        return Y

    def forward(self, image, full=False):
        # Extract Luma channel (Y from YCbCr) from RGB data.
        Y = self.rgb2ycbcr(image)
        # create a mask to isolate light sources
        mask = torch.sigmoid(self.b*(Y - self.a))
        # Multiply the image by the mask to isolate
        # light sources
        img_light = torch.mul(image, mask)
        return img_light


class LinearConv(nn.Module):
    """
    Apply a Learnable Linear Convolution to the input image

    Args: 
        - kernel_size (int): size of the convolution kernel.
    """
    def __init__(self, kernel_size=7):
        super(LinearConv, self).__init__()
        
        # Define Kernels for x and y
        kernel_x = torch.zeros((1, 1, 1, kernel_size))
        kernel_y = torch.zeros((1, 1, kernel_size, 1))

        # Initialise the kernels as dirac delta impulses
        kernel_x[:, :, :, kernel_size//2] = 1
        kernel_y[:, :, kernel_size//2, :] = 1
        # Add some light noise to the kernel weights
        noise_x = torch.normal(0, 0.01, size=(1, 1, 1, kernel_size))
        noise_y = torch.normal(0, 0.01, size=(1, 1, kernel_size, 1))
        kernel_x += noise_x
        kernel_y += noise_y

        # Define Kernels as trainable parameters.
        self.kernel_x = nn.parameter.Parameter(kernel_x)
        self.kernel_y = nn.parameter.Parameter(kernel_y)

    def forward(self, image):

        kernel_x = self.kernel_x
        kernel_y = self.kernel_y
        
        # Normalise kernels
        kernel_x = kernel_x / kernel_x.sum()
        kernel_y = kernel_y / kernel_y.sum()

        kernel_x = kernel_x.expand(image.shape[-3], 1, kernel_x.shape[2], kernel_x.shape[3])
        kernel_y = kernel_y.expand(image.shape[-3], 1, kernel_y.shape[2], kernel_y.shape[3])
        # Apply horizontal kernel
        Y = F.conv2d(image, kernel_x, padding='same', groups=image.shape[-3])
        # Apply vertical kernel
        out = F.conv2d(Y, kernel_y, padding='same', groups=image.shape[-3])

        return out

class HighFreq(nn.Module):
    """
    Introduce noise in the input image by replicating the noise model in a camera. 
    The shader learns the optimal amount of noise added.
    """
    def __init__(self):
        super(HighFreq, self).__init__()
    
        self.gain = nn.parameter.Parameter(torch.tensor([-10.0]))
        self.sigma = nn.parameter.Parameter(torch.tensor([0.001]))
        self.register_buffer("mu", torch.tensor([0.0]))

    def forward(self, image):
        img_shape = image.shape

        # The gain is encoded in this way to prevent NaNs if self.gain 
        # is too small.
        gain = torch.pow(2, self.gain)
        gamma_expand = torch.pow(torch.clamp(image, min=1e-7), 2.2)
        gamma_expand_gain = gamma_expand / gain 

        # Synthesise approximated Poisson Noise.
        poisson = torch.clamp(torch.randn(
            img_shape[0], img_shape[1], img_shape[2], img_shape[3], device=image.device), min=0)
        new_image = torch.mul(torch.sqrt(gamma_expand_gain), poisson) + gamma_expand_gain
        
        # Compress input image back.
        new_image *= gain   
        new_image = torch.pow(new_image, 1/2.2)

        # Add Normally distributed noise
        high_freq_noise = torch.randn(
            img_shape[0], img_shape[1], img_shape[2], img_shape[3], device=image.device)
        high_freq_noise = (high_freq_noise * self.sigma) + self.mu 
        result = new_image + high_freq_noise

        return result

class HighFreqCMAP(nn.Module):
    """
    Introduce noise in the input image by replicating the noise model in a camera. 
    The shader learns the optimal amount of noise added. This one also returns 
    the colour mapping output, which is needed when colour mapping constraint is
    applied.
    """
    def __init__(self):
        super(HighFreqCMAP, self).__init__()

        self.gain = nn.parameter.Parameter(torch.tensor([-10.0]))
        self.sigma = nn.parameter.Parameter(torch.tensor([0.001]))
        self.register_buffer("mu", torch.tensor([0.0]))

    def forward(self, x):
        image = x[0]
        cmap = x[1]
        img_shape = image.shape
        
        # The gain is encoded in this way to prevent NaNs if self.gain 
        # is too small.
        gain = torch.pow(2, self.gain)
        gamma_expand = torch.pow(torch.clamp(image, min=1e-7), 2.2)
        gamma_expand_gain = gamma_expand / gain 

        # Synthesise approximated Poisson Noise.
        poisson = torch.clamp(torch.randn(
            img_shape[0], img_shape[1], img_shape[2], img_shape[3], device=image.device), min=0)
        new_image = torch.mul(torch.sqrt(gamma_expand_gain), poisson) + gamma_expand_gain

        # Compress input image back.
        new_image *= gain
        new_image = torch.pow(new_image, 1/2.2)

        # Add Normally distributed noise
        high_freq_noise = torch.randn(
            img_shape[0], img_shape[1], img_shape[2], img_shape[3], device=image.device)
        high_freq_noise = (high_freq_noise * self.sigma) + self.mu 
        result = new_image + high_freq_noise

        return result, cmap

def define_shaders(shaders="cmap"):
    def define_cmap():
        cmap = ColourMapping()
        return cmap

    def define_bloom_multires():
        bloom = BloomMultiRes(blur_magnitude=False, return_cmap=False)
        return bloom
    
    def define_bloom_multires_return_cmap():
        bloom = BloomMultiRes(blur_magnitude=False, return_cmap=True)
        return bloom

    def define_lens_blur_spatial_7():
        lens = LinearConv(kernel_size=7)
        return lens

    def define_lens_blur_spatial_5():
        lens = LinearConv(kernel_size=5)
        return lens
    
    def define_high_freq():
        highfreq = HighFreq()
        return highfreq

    def define_high_freq_cmap():
        highfreq = HighFreqCMAP()
        return highfreq

    shaders_dict = {
        'cmap':define_cmap,
        'bloom_multires':define_bloom_multires,
        'bloom_multires_return_cmap':define_bloom_multires_return_cmap,
        'lens_spatial_7':define_lens_blur_spatial_7,
        'lens_spatial_5':define_lens_blur_spatial_5,
        'highfreq':define_high_freq,
        'highfreq_cmap':define_high_freq_cmap
    }

    if shaders is None:
        return None

    if shaders not in shaders_dict:
        raise TypeError('shader key {0} not defined.'.format(shaders))

    shaders_list = shaders_dict[shaders]()
    return shaders_list

