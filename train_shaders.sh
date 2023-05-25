#!/bin/bash

# Copyright:    Copyright (c) Imagination Technologies Ltd 2023
# License:      MIT (refer to the accompanying LICENSE file)
# Author:       AI Research, Imagination Technologies Ltd
# Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


CUDA_VISIBLE_DEVICES=0 python3 ./train_shaders/train_shaders.py \
--image_size=291 \
--batch_size=8 \
--n_epochs=2000 \
--num_workers=4 \
--pipeline_specs="./pipeline_specs/pipeline.csv" \
--results_dir="./results_test" \
--render_train_dir="/datasets/playing_for_data/images" \
--real_train_dir="/datasets/cityscapes/leftImg8bit/" \
--target_dataset="cityscapes" \
--discriminator="noise" 
