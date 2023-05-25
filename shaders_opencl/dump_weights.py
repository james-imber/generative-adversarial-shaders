# Copyright:    Copyright (c) Imagination Technologies Ltd 2023
# License:      MIT (refer to the accompanying LICENSE file)
# Author:       AI Research, Imagination Technologies Ltd
# Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


import sys
sys.path.insert(1, "../train_shaders")

from shaders import *
from deep_shaders import *
from pipeline import *
import kernel_str

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import argparse
import subprocess

def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--specs_file", default="./full_p_onecycle_all.csv", type=str, help="Specifies path to specs file to be used to load weights.")
    parser.add_argument("--save_json", default=False, action="store_true", help="If True, lsave loaded weights into json file.")
    parser.add_argument("--json_out", default="ocl_weights.json", type=str, help="Specifies name of saved json file")
    parser.add_argument("--load_json", default=False, action="store_true", help="If True, load weights from json file.")
    parser.add_argument("--json_path", default="", type=str, help="if load_json True, specifies path from which json file is loaded.")

    args = parser.parse_args()
    return args 

def extract_weights(checkpoint):
    mapping_dict = {}

    for weight in checkpoint['state_dict']:
        weight_split = weight.split('.')
        if len(weight_split) == 3:
            mapping_dict[weight_split[2]] = checkpoint['state_dict'][weight]
        else:
            mapping_dict[weight_split[2]+'.'+weight_split[3]] = checkpoint['state_dict'][weight]

    return mapping_dict

def extract_shaders_weights(p_specs):

    df_shaders = pd.read_csv(p_specs)
    paths = df_shaders['weights_dir'].drop_duplicates().to_list()
    
    checkpoint = {}

    for path in paths:
        checkpoint[path] = torch.load(path)

    weights = {}
    for index, row in df_shaders.iterrows():
        shader = define_shaders(row['shader'])
        mapping_dict = extract_weights(checkpoint[row['weights_dir']])

        weight_mapping = {}
        for name, T in shader.named_parameters():
            weight_mapping[name] = mapping_dict[name]

        weights[row['shader']] = weight_mapping

    return weights

def parse_lens_blur_params(weight_mapping):
    if 'lens_spatial_5' not in weight_mapping:
        return

    kernel_x = weight_mapping['lens_spatial_5']['kernel_x'][0][0][0]
    kernel_x = kernel_x / kernel_x.sum()

    weight_mapping['lens_spatial_5']['kernel_x'] = kernel_x.tolist()

    kernel_y = weight_mapping['lens_spatial_5']['kernel_y'].permute(0, 1, 3, 2)[0][0][0]
    kernel_y = kernel_y / kernel_y.sum()

    weight_mapping['lens_spatial_5']['kernel_y'] = kernel_y.tolist()

    return

def parse_colour_mapping_params(weight_mapping):
    if 'cmap' not in weight_mapping:
        return
    
    weight_mapping['cmap']['transform'] = weight_mapping['cmap']['transform'][0].tolist()
    weight_mapping['cmap']['scaling_vector'] = weight_mapping['cmap']['scaling_vector'][0].permute(1, 0)[0].tolist()
    return

def parse_bloom_params(weight_mapping):
    def calculate_kernel_weights(sigma, magnitude):
        half_kernel_size = 5 // 2
        x = torch.linspace(-half_kernel_size, half_kernel_size, steps=5).to(device=sigma.device)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))

        kernel1d = pdf / pdf.sum()
        kernel1d *= magnitude
        kernel1d = kernel1d.tolist()

        kernel1d = [i if i >= 10**-4 else 0.0 for i in kernel1d]
        return kernel1d

    if 'bloom' not in weight_mapping:
        return

    weight_mapping['bloom']['exposure'] = weight_mapping['bloom']['exposure'].item()

    res = [1, 2, 4, 8]

    for i in res:
        weight_mapping['bloom'][f'mask_{i}.a'] = weight_mapping['bloom'][f'mask_{i}.a'].item()
        weight_mapping['bloom'][f'mask_{i}.b'] = weight_mapping['bloom'][f'mask_{i}.b'].item()
        weight_mapping['bloom'][f'mask_{i}.Y_weight'] = weight_mapping['bloom'][f'mask_{i}.Y_weight'].permute(1, 0)[0].tolist()
        
        kernel_x = calculate_kernel_weights(
            weight_mapping['bloom'][f'blur_{i}.sigma_x'], weight_mapping['bloom'][f'blur_{i}.magnitude'])
        weight_mapping['bloom'][f'blur_{i}.sigma_x'] = weight_mapping['bloom'][f'blur_{i}.sigma_x'].item()
        weight_mapping['bloom'][f'blur_{i}.kernel_x'] = kernel_x

        kernel_y = calculate_kernel_weights(
            weight_mapping['bloom'][f'blur_{i}.sigma_y'], weight_mapping['bloom'][f'blur_{i}.magnitude'])
        weight_mapping['bloom'][f'blur_{i}.sigma_y'] = weight_mapping['bloom'][f'blur_{i}.sigma_y'].item()
        weight_mapping['bloom'][f'blur_{i}.kernel_y'] = kernel_y

        weight_mapping['bloom'][f'blur_{i}.magnitude'] = weight_mapping['bloom'][f'blur_{i}.magnitude'].item()
    
    return

def parse_noise_params(weight_mapping):
    if 'highfreq' not in weight_mapping:
        return

    weight_mapping['highfreq']['gain'] = weight_mapping['highfreq']['gain'].item()
    weight_mapping['highfreq']['sigma'] = weight_mapping['highfreq']['sigma'].item()
    weight_mapping['highfreq']['mu'] = weight_mapping['highfreq']['mu'].item()

    return

def parse_weights(weight_mapping):
    parse_lens_blur_params(weight_mapping)    
    parse_colour_mapping_params(weight_mapping)    
    parse_bloom_params(weight_mapping)
    parse_noise_params(weight_mapping)
    return

def write_header_file(weight_mapping):
    lens_blur_params = """
// Lens Blur, Colour Mapping and Full Res Bloom

#define KERNEL_X_LENS_BLUR {{{0}, {1}, {2}, {3}, {4}}}
#define KERNEL_Y_LENS_BLUR {{{5}, {6}, {7}, {8}, {9}}}
    """.format(
        weight_mapping['lens_spatial_5']['kernel_x'][0], 
        weight_mapping['lens_spatial_5']['kernel_x'][1], 
        weight_mapping['lens_spatial_5']['kernel_x'][2], 
        weight_mapping['lens_spatial_5']['kernel_x'][3], 
        weight_mapping['lens_spatial_5']['kernel_x'][4], 
        weight_mapping['lens_spatial_5']['kernel_y'][0], 
        weight_mapping['lens_spatial_5']['kernel_y'][1], 
        weight_mapping['lens_spatial_5']['kernel_y'][2], 
        weight_mapping['lens_spatial_5']['kernel_y'][3], 
        weight_mapping['lens_spatial_5']['kernel_y'][4]
    )

    cmap_params = """
#define COLOUR_TRANSFORM {{{{{0}, {1}, {2}}},{{{3}, {4}, {5}}},{{{6}, {7}, {8}}}}}    
#define COLOUR_TRANSLATION {{{9}, {10}, {11}}}
""".format(
    weight_mapping['cmap']['transform'][0][0],
    weight_mapping['cmap']['transform'][0][1],
    weight_mapping['cmap']['transform'][0][2],
    weight_mapping['cmap']['transform'][1][0],
    weight_mapping['cmap']['transform'][1][1],
    weight_mapping['cmap']['transform'][1][2],
    weight_mapping['cmap']['transform'][2][0],
    weight_mapping['cmap']['transform'][2][1],
    weight_mapping['cmap']['transform'][2][2],
    weight_mapping['cmap']['scaling_vector'][0],
    weight_mapping['cmap']['scaling_vector'][1],
    weight_mapping['cmap']['scaling_vector'][2]
)

    res = [1, 2, 4, 8]
    res_dict = {
        1:"FULL", 
        2:"HALF", 
        4:"QUARTER", 
        8:"EIGHTH"
    }

    bloom_params = ""

    for i in res:
        bloom_params += """
#define Y_WEIGHT_{0}_RES {{{1}, {2}, {3}}} 
#define SIG_SHIFT_{0}_RES {4}
#define SIG_SCALE_{0}_RES {5}
#define KERNEL_X_BLOOM_{0}_RES {{{6}, {7}, {8}, {9}, {10}}}
#define KERNEL_Y_BLOOM_{0}_RES {{{11}, {12}, {13}, {14}, {15}}}
    """.format(
            res_dict[i],
            weight_mapping['bloom'][f'mask_{i}.Y_weight'][0],
            weight_mapping['bloom'][f'mask_{i}.Y_weight'][1],
            weight_mapping['bloom'][f'mask_{i}.Y_weight'][2],
            weight_mapping['bloom'][f'mask_{i}.a'],
            weight_mapping['bloom'][f'mask_{i}.b'],
            weight_mapping['bloom'][f'blur_{i}.kernel_x'][0],
            weight_mapping['bloom'][f'blur_{i}.kernel_x'][1],
            weight_mapping['bloom'][f'blur_{i}.kernel_x'][2],
            weight_mapping['bloom'][f'blur_{i}.kernel_x'][3],
            weight_mapping['bloom'][f'blur_{i}.kernel_x'][4],
            weight_mapping['bloom'][f'blur_{i}.kernel_y'][0],
            weight_mapping['bloom'][f'blur_{i}.kernel_y'][1],
            weight_mapping['bloom'][f'blur_{i}.kernel_y'][2],
            weight_mapping['bloom'][f'blur_{i}.kernel_y'][3],
            weight_mapping['bloom'][f'blur_{i}.kernel_y'][4]
        )

    blend_params = """
#define EXPOSURE {0}
#define GAIN {1}
#define SIGMA {2}
#define MU {3}
    """.format(
        weight_mapping['bloom']['exposure'],
        weight_mapping['highfreq']['gain'],
        weight_mapping['highfreq']['sigma'],
        weight_mapping['highfreq']['mu']
    )

    # print(lens_blur_params + cmap_params + bloom_params + blend_params)
    with open("./cl_files/params_dump.h", "w") as f:
        f.write(lens_blur_params)
        f.write(cmap_params)
        f.write(bloom_params)
        f.write(blend_params)

    return

def main():

    args = arguments()

    if args.load_json:
        with open(args.json_path, "r") as json_file:
            weight_mapping = json.loads(json_file.read())

        print(f"weights loaded from json {args.json_path}")
    else:
        weight_mapping = extract_shaders_weights(args.specs_file)
        parse_weights(weight_mapping)

        print(f"weights loaded from specs file {args.specs_file}")

        if args.save_json:
            with open(args.json_out, "w") as json_file:
                json.dump(weight_mapping, json_file, indent=4)

            print(f"weights saved in {args.json_out}")

    write_header_file(weight_mapping)

    return 

if __name__ == "__main__":
    main()
