# Copyright:    Copyright (c) Imagination Technologies Ltd 2023
# License:      MIT (refer to the accompanying LICENSE file)
# Author:       AI Research, Imagination Technologies Ltd
# Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


import argparse

def train_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int, help="batch size used during training")
    parser.add_argument("--n_epochs", default=2000, type=int, help="number of training epochs")
    parser.add_argument("--image_size", default=291, type=int, help="image crop size")
    parser.add_argument("--render_train_dir", default="GTA", type=str, help="directory from which render images are loaded for training")
    parser.add_argument("--real_train_dir", default="cityscapes", type=str, help="directory from which real world images are loaded for training")
    parser.add_argument("--pipeline_specs", default="pipeline_specs.csv", type=str, help="specify path to file describing shaders needed to train the pipeline.")
    parser.add_argument("--target_dataset", default="cityscapes", type=str, help="Dataset used as target")
    parser.add_argument("--discriminator", default=None, type=str, help="Specify which type of discriminator to use.")
    parser.add_argument("--resume_training", default=False, action="store_true", help="If True, resume training from a given epoch.")
    parser.add_argument("--pretrained_dir", default="", type=str, help="directory from which pretrained discriminator and pipeline are loaded from")
    parser.add_argument("--resume_from", default=0, type=int, help="Epoch number from which training is resumed")
    parser.add_argument("--num_workers", default=4, type=int, help="number of cpu workers to use to load data")
    parser.add_argument("--results_dir", default="./pipeline_results", type=str, help="path where training results are stored")
    parser.add_argument("--colour_map_constraint", default=False, action="store_true", help="If True, apply constraint to colour mapping to limit output between 0 and 1.")
    args = parser.parse_args()
    return args
