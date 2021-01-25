import os
import sys
import time
import numpy as np
import torchlayers as tl
from tqdm import tqdm
from scipy.stats import norm
from torchsummary import summary

from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils

from args import TrainArgParser
from utils import sample_noise, load_models, load_optimizer, visualize

from pdb import set_trace as st

def train(args):
    print(args.name)

    # Load dataset/dataloader
    dataset = Dataset(args.dataset_name, args.batch_size)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=True)
    
    # Fixed latent for visualizing training only
    fixed_zz = sample_noise(args.viz_batch_size, args.device)
   
    # Load models
    generator, discriminator = load_models(args)

    # Loss and optimizers
    criterion = nn.BCEWithLogitsLoss()
    optimizer_generator, optimizer_discriminator = load_optimizer(args, [generator, discriminator])

    ckpt_paths = []

    discriminator = discriminator.to(args.device)
    generator = generator.to(args.device)

    # Initialize labels 
    real_label = torch.ones(batch_size_i, 1).float().cuda()
    fake_label = torch.zeros(batch_size_i, 1).float().cuda()
    realism_labels = torch.cat((real_label, fake_label), dim=0) # stacked labels
    
    # Training loop
    num_images = len(dataset)
    for epoch in range(args.num_epochs):
        for i, real in enumerate(tqdm(loader)):
            batch_size_i = len(real)
            step = epoch * num_images + i
            if step == 0:
                save_path = Path(args.viz_dir) / f'reals.png'
                visualize(real[:args.viz_batch_size], save_path)

            real = real.to(args.device).float()

            # Train generator
            generator.zero_grad()
            z = sample_noise(batch_size_i, args.device)
            fake = generator(z)
            decision_fake_g = discriminator(fake)

            error_generator = criterion(decision_fake_g, real_label)
            error_generator.backward()
            
            optimizer_generator.step()
           
            if step % args.step_train_discriminator == 0:

                # Train discriminator
                discriminator.zero_grad()
                decision_real = discriminator(real, real_label)
                
                with torch.no_grad():
                    z = sample_noise(batch_size_i, args.device)
                    fake = generator(z)
                decision_fake_d = discriminator(fake, fake_label)

                # Stacked decisions + stacked loss backprop
                decisions = torch.cat((decision_real, decision_fake_d), dim=0).cuda()
                error_discriminator = criterion(decisions, realism_labels)
                error_discriminator.backward()

                optimizer_discriminator.step()

                update_message = f'[{epoch}/{args.num_epochs}]\t'
                update_message += f'Loss disc: {error_discriminator.item():.4f}\tLoss gen: {error_generator.item():.4f}\t'
                update_message += f'D(real): {decision_real.mean().item():.2f}\t'
                update_message += f'D(G(z)): {decision_fake_g.mean().item():.2f} and {decision_fake_d.mean().item():.2f}\t'
                update_message += f'{args.name}\t'
                print(update_message)

            # Save every epoch (can add viz step as arg too)
            if step % num_images == 0:
                fakes = []
                for iz, fixed_z in enumerate(fixed_zs):
                    with torch.no_grad():
                        fake = generator(fixed_z).detach().cpu()
                        fakes.append(fake)
                fakes_concat = torch.cat(fakes, 0)
                save_path = Path(args.viz_dir) / f'step_{step}.png'
                visualize(fakes_concat, save_path)

                # Evaluation, save ckpt of generator only for now
                ckpt_dict = {
                    'ckpt_info': {'step': step},
                    'model_name': generator.module.__class__.__name__,
                    'model_args': generator.module.args_dict(),
                    'model_state': generator.to('cpu').state_dict(),
                    'optimizer': optimizer_generator.state_dict(),
                }
                generator.to(args.device)

                ckpt_path = Path(args.ckpt_dir) / f'step_{step}.pth.tar'
                torch.save(ckpt_dict, ckpt_path)
                
                ckpt_paths.append(ckpt_path)
                if len(ckpt_paths) > args.max_ckpts:
                    oldest_ckpt = ckpt_paths.pop(0)
                    os.remove(oldest_ckpt)

    
    fakes = []
    for iz, fixed_z in enumerate(fixed_zs):
        with torch.no_grad():
            fake = generator(fixed_z).detach().cpu()
            fakes.append(fake)
    fakes_concat = torch.cat(fakes, 0)
    save_path = Path(args.viz_dir) / f'step_{step}.png'
    visualize(fakes_concat, save_path)
    
    ckpt_path = Path(args.ckpt_dir) / f'step_{step}.pth.tar'
    torch.save(ckpt_dict, ckpt_path)
            
    ckpt_paths.append(ckpt_path)
    if len(ckpt_paths) > args.max_ckpts:
        oldest_ckpt = ckpt_paths.pop(0)
        os.remove(oldest_ckpt)


if __name__ == "__main__":
    parser = TrainArgParser()
    args_ = parser.parse_args()
    train(args_)
