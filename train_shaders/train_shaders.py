# Copyright:    Copyright (c) Imagination Technologies Ltd 2023
# License:      MIT (refer to the accompanying LICENSE file)
# Author:       AI Research, Imagination Technologies Ltd
# Paper:        Generative Adversarial Shaders for Real-Time Realism Enhancement


from shaders import *
from discriminator import *
from loss import *
from pipeline import *
import utils

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms.functional import center_crop

import os
from PIL import Image
from config import train_config
from data import TrainImageDataset
import numpy as np

def main():

    # Define training arguments
    args = train_config()
    
    # Build training dataset
    train_dataloader = build_dataset(args)

    # Build pipeline from specs file
    pipeline = build_pipeline(args.pipeline_specs)

    # Define discriminator
    discriminator = build_discriminator(args.discriminator)

    # Define optimisers
    p_optimizer, d_optimizer = define_optimizers(pipeline, discriminator)

    if args.resume_training:
        p_checkpoint = torch.load(os.path.join(args.pretrained_dir, f"g_epoch_{args.resume_from}.pth.tar"))
        pipeline.load_state_dict(p_checkpoint['state_dict'])                            
        p_optimizer.load_state_dict(p_checkpoint['optimizer'])      

        d_checkpoint = torch.load(os.path.join(args.pretrained_dir, f"d_epoch_{args.resume_from}.pth.tar")) 
        discriminator.load_state_dict(d_checkpoint['state_dict'])
        d_optimizer.load_state_dict(d_checkpoint['optimizer'])                     
    
    # Define loss criterions
    dummy_batch = torch.zeros((8, 3, args.image_size, args.image_size))
    # NB: We crop out 35 pixels around the edges of the pipeline output
    # as this is half the receptive field of the discriminator.
    dummy_batch = center_crop(dummy_batch, (args.image_size - 35))
    out_dummy = discriminator(dummy_batch.to(dtype=torch.float32, device=torch.device('cuda', 0)))
    if isinstance(out_dummy, list):
        d_output_size = args.batch_size
    else:
        d_output_size = out_dummy.shape
    GAN_criterion = define_loss_criterion(d_output_size)

    # Set up directory to store trained model at different epochs
    trained_weights_dir = args.results_dir + "/weights"
    if not os.path.exists(trained_weights_dir):
        os.makedirs(trained_weights_dir)

    # Set up writer to keep track of models progress
    writer = SummaryWriter(args.results_dir + "/logs")

    # Create directory to save images during training
    image_path = args.results_dir + "/images"
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    render_test = Image.open(utils.get_file_names(args.render_train_dir)[0])
    render_test = utils.image_to_tensor(render_test).unsqueeze(0)

    epochs = args.n_epochs

    # Repeat training for n epochs
    for epoch in range(epochs):

        batch_losses = train(
                             train_dataloader,
                             pipeline,
                             discriminator,
                             p_optimizer,
                             d_optimizer,
                             GAN_criterion,
                             (args.image_size - 35),
                             args.colour_map_constraint
                            )

        print('Epoch [%d/%d]\tLoss D: %.4f\tLoss G(adversarial): %.4f\tLoss G: %.4f\t'
                  % (epoch + 1, epochs,
                     batch_losses[0], batch_losses[1], batch_losses[2]))
        
        # Update tfevents file
        writer.add_scalar("Train/D_Loss", batch_losses[0].item(), args.resume_from + epoch+1)
        writer.add_scalar("Train/G_Loss_adv", batch_losses[1].item(), args.resume_from + epoch+1)
        writer.add_scalar("Train/G_Loss", batch_losses[2].item(), args.resume_from + epoch+1)

        if (epoch + 1) % 10 == 0:
            torch.save({"epoch": args.resume_from + epoch + 1,
                        "state_dict": discriminator.state_dict(),
                        "optimizer": d_optimizer.state_dict()},
                        os.path.join(trained_weights_dir, "d_epoch_{0}.pth.tar".format(args.resume_from + epoch + 1)))
            torch.save({"epoch": args.resume_from + epoch + 1,
                        "state_dict": pipeline.state_dict(),
                        "optimizer": p_optimizer.state_dict()},
                        os.path.join(trained_weights_dir, "g_epoch_{0}.pth.tar".format(args.resume_from + epoch + 1)))
            
            with torch.no_grad():
                shaded = pipeline(render_test.to(device=torch.device('cuda', 0)))
            
            if args.colour_map_constraint:
                save_image(shaded[0], image_path + "/epoch_{0}.png".format(epoch + 1))
            else:
                save_image(shaded, image_path + "/epoch_{0}.png".format(epoch + 1))

    # Save last epoch in case number of epochs isn't a number divisible by 10
    if epochs % 10 != 0:
        torch.save({"epoch": args.resume_from + epoch + 1,
                    "state_dict": discriminator.state_dict(),
                    "optimizer": d_optimizer.state_dict()},
                    os.path.join(trained_weights_dir, "d_epoch_{0}.pth.tar".format(args.resume_from + epoch + 1)))
        torch.save({"epoch": args.resume_from + epoch + 1,
                    "state_dict": pipeline.state_dict(),
                    "optimizer": p_optimizer.state_dict()},
                    os.path.join(trained_weights_dir, "g_epoch_{0}.pth.tar".format(args.resume_from + epoch + 1)))
        
        with torch.no_grad():
            shaded = pipeline(render_test.to(device=torch.device('cuda', 0)))

        if args.colour_map_constraint:
            save_image(shaded[0], image_path + "/epoch_{0}.png".format(epoch + 1))
        else:
            save_image(shaded, image_path + "/epoch_{0}.png".format(epoch + 1))

    return

def train(train_dataloader, pipeline, discriminator, p_optimizer,
          d_optimizer, GAN_criterion, crop_size, colour_map_constraint):

    d_losses = []
    gan_losses = []
    p_losses = []

    for _, data in enumerate(train_dataloader):

        render = data["render"].to(device=torch.device('cuda', 0))
        real = data["real"].to(device=torch.device('cuda', 0))

        discriminator.zero_grad()

        if colour_map_constraint:
            fake = pipeline(render)[0]
        else:
            fake = pipeline(render)

        # Get discriminator predictions
        # crop borders if shader generates artifacts at image edges
        real_crop = center_crop(real, crop_size)
        fake_crop = center_crop(fake, crop_size)
        d_output_r = discriminator(real_crop)
        d_output_f = discriminator(fake_crop)

        # Get discriminator loss
        d_loss = GAN_criterion(d_output_r, d_output_f)

        # Update discriminator
        d_loss.backward()
        d_optimizer.step()

        # Set pipeline for update
        pipeline.zero_grad()

        # Generate new images from the pipeline
        if colour_map_constraint:
            fake, cmap = pipeline(render)
        else:
            fake = pipeline(render)

        # Get discriminator predictions
        # crop borders if shader generates artifacts at image edges
        real_crop = center_crop(real, crop_size)
        fake_crop = center_crop(fake, crop_size)

        d_output_r = discriminator(real_crop)
        d_output_f = discriminator(fake_crop)

        # Get gan loss
        gan_l = GAN_criterion(d_output_f, d_output_r)
        p_loss = gan_l

        if colour_map_constraint:
            cmap_loss = torch.mean(torch.clamp(cmap.amax(dim=(2, 3)), min=1.0) - torch.clamp(cmap.amin(dim=(2, 3)), max=0.0)) * 0.001
            p_loss += cmap_loss        

        # Update pipeline
        p_loss.backward()
        p_optimizer.step()

        d_losses.append(d_loss.detach().item())
        gan_losses.append(gan_l.detach().item())

        if colour_map_constraint:
            p_losses.append(p_loss.detach().item() + cmap_loss.detach().item())
        else:
            p_losses.append(p_loss.detach().item())

    return [np.mean(d_losses), np.mean(gan_losses), np.mean(p_losses)]

def build_dataset(args):
    """
    Utility function to build the datasets
    """

    # Include depth if needed for training, else just use real
    # and rendered images.
    train_dataset = TrainImageDataset(
        args.render_train_dir, 
        args.real_train_dir, 
        args.image_size, 
        args.target_dataset
    )
    
    # Define the dataloader
    train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True,
            drop_last=True)

    return train_dataloader

def build_pipeline(pipeline_specs):
    """
    Utility function to build the pipeline
    """
    pipeline = GraphicsPipeline(pipeline_specs)

    pipeline.to(
        device=torch.device("cuda", 0),
    )

    return pipeline

def build_discriminator(regularisation):
    """
    Utility function to build the discriminator
    """
    def patch_gan():
        print("Using default PatchGAN.")
        return PatchGAN()

    def patch_gan_noise():
        print("Using PatchGAN with instance noise.")
        return PatchGANWithNoise()

    def patch_gan_noise_batchnorm():
        print("Using PatchGAN with instance noise and Batch Normalisation.")
        return PatchGANWithNoise(normalisation='BatchNorm')

    discriminators = {
        'path_gan':patch_gan,
        'noise':patch_gan_noise,
        'noise_batchnorm':patch_gan_noise_batchnorm
    }

    if regularisation not in discriminators:
        discriminator = patch_gan()
    else:
        discriminator = discriminators[regularisation]()

    discriminator.to(
        device=torch.device("cuda", 0),
    )

    return discriminator

def define_optimizers(pipeline, discriminator):
    """
    Instantiate optimizers for Adversarial Training.
    """

    p_optimizer = optim.Adam(
        pipeline.parameters(), 
        lr=1e-4, betas=(0.9, 0.999)
    )

    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=1e-4, betas=(0.9, 0.999)
    )

    return p_optimizer, d_optimizer

def define_loss_criterion(prediction_size):
        
    GAN_criterion = GANLoss(prediction_size)

    GAN_criterion.to(
        device=torch.device("cuda", 0),
        memory_format=torch.channels_last
    )

    return GAN_criterion

if __name__ == "__main__":
    main()

