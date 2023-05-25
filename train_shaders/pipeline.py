# Copyright:    Copyright (c) Imagination Technologies Ltd 2023
# License:      MIT (refer to the accompanying LICENSE file)
# Author:       AI Research, Imagination Technologies Ltd
# Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


import torch 
from torch import nn
import pandas as pd
from shaders import *

class GraphicsPipeline(nn.Module):
    """
    Define Torch object for a learnable shader pipeline.
    Args:
        - p_specs [str]: path to csv file specifying which shaders to use
          and in which order.
    """
    def __init__(self, p_specs, verbose=True):
        super(GraphicsPipeline, self).__init__()

        # Loads shaders to be used in a list
        pipeline = load_shaders(p_specs, verbose)

        # Define Pipeline block 
        self.pipeline = nn.Sequential(*pipeline)

    def forward(self, x):
        # Pass input batch through pipeline
        y = self.pipeline(x)
        
        if isinstance(y, tuple):
            return torch.clamp(y[0], min=0, max=1), y[1]
        
        return torch.clamp(y, min=0, max=1)

def load_shaders(p_specs, verbose):
    df_shaders = pd.read_csv(p_specs)
    
    pipeline = []

    for index, row in df_shaders.iterrows():
        if row['shader_type'] == 'pretrained':
            
            if verbose: print(f"Shader {index}: {row['shader']}, pretrained")

            shader = define_shaders(row['shader'])
            checkpoint = torch.load(row['weights_dir'])
            shader = load_weights(shader, checkpoint)

            pipeline.append(shader)

        elif row['shader_type'] == 'frozen':

            if verbose: print(f"Shader {index}: {row['shader']}, frozen")

            shader = define_shaders(row['shader'])
            checkpoint = torch.load(row['weights_dir'])
            shader = load_weights(shader, checkpoint)

            shader.eval()
            for params in shader.parameters():
                params.requires_grad = False

            pipeline.append(shader)
        else:

            if verbose: print(f"Shader {index}: {row['shader']}, train")

            pipeline.append(define_shaders(row['shader']))
    
    return pipeline

def load_weights(shader, checkpoint):
    # for i in checkpoint:
    shader_state_dict = shader.state_dict()
    mapping_dict = {}
    for weight in checkpoint['state_dict']:
        weight_split = weight.split('.')
        if len(weight_split) == 3:
            mapping_dict[weight_split[2]] = checkpoint['state_dict'][weight]
        else:
            mapping_dict[weight_split[2]+'.'+weight_split[3]] = checkpoint['state_dict'][weight]
    
    for name, T in shader.named_parameters():
        shader_state_dict[name].copy_(mapping_dict[name])

    shader.load_state_dict(shader_state_dict)
    return shader
