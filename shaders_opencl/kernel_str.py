# Copyright:    Copyright (c) Imagination Technologies Ltd 2023
# License:      MIT (refer to the accompanying LICENSE file)
# Author:       AI Research, Imagination Technologies Ltd
# Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


lens_blur_x_kernel = "float kernel_h[KERNEL_SIZE] = {{{0}, {1}, {2}, {3}, {4}}};\n"

lens_blur_y_kernel = "float kernel_v[KERNEL_SIZE] = {{{0}, {1}, {2}, {3}, {4}}};\n"

colour_map_transform = """
__constant float colour_transform[ARRAY_SIZE][ARRAY_SIZE] = {{
    {{{0}, {1}, {2}}},
    {{{3}, {4}, {5}}},
    {{{6}, {7}, {8}}}
}};
"""

colour_map_translate = """
__constant float colour_translation[ARRAY_SIZE] = {{{0}, {1}, {2}}};
"""

light_mask_full_res = """
    float Y_weight[3] = {{{}, {}, {}}};
    float sig_shift = {};
    float sig_scale = {};
""" 
