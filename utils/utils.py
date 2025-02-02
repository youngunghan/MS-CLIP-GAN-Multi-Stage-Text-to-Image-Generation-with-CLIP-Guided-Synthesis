import os
import sys
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def normalize(feature, dim=-1):
    return F.normalize(feature, p=2, dim=dim)

def seed_fix(int):
    # PyTorch
    torch.manual_seed(int)
    torch.cuda.manual_seed(int)
    torch.cuda.manual_seed_all(int) # for multi-GPU

    # CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Numpy
    np.random.seed(int)

    # Random
    random.seed(int)

def weight_init(layer):
    # Do NOT modify
    if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)

    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias.data, val=0)

    elif isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, val=0.0)

def mkdirs(paths):
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)
    
def save_checkpoint(args, g: torch.nn.Module, d_lst: List[torch.nn.Module], 
                optim_g, optim_d_lst, epoch: int, num_stage: int) -> None:
    # """Save generator and discriminator models"""
    # Save generator
    generator_state = {
        'model': g.state_dict(),
        'optimizer': optim_g.state_dict(),
        'epoch': epoch,
        'num_stage': num_stage
    }
    torch.save(generator_state, 
              os.path.join(args.checkpoint_path, f"epoch_{epoch}_Gen.pt"))
    
    # Save discriminators
    for i, (disc, optim_d) in enumerate(zip(d_lst, optim_d_lst)):
        discriminator_state = {
            'model': disc.state_dict(),
            'optimizer': optim_d.state_dict(),
            'epoch': epoch,
            'num_stage': num_stage
        }
        torch.save(discriminator_state, 
                  os.path.join(args.checkpoint_path, f"epoch_{epoch}_Dis_{i}.pt"))
    
    print(f'Saved models to {args.checkpoint_path}')
    
def load_checkpoint(args, g: torch.nn.Module, d_lst: List[torch.nn.Module],
               optim_g, optim_d_lst: List[torch.optim.Optimizer], checkpoint_path, epoch: int) -> tuple:
    """Load generator and discriminator models with their optimizers"""
    device = next(g.parameters()).device
    
    # Load generator
    gen_path = os.path.join(checkpoint_path, f"epoch_{epoch}_Gen.pt")
    if os.path.exists(gen_path):
        gen_state = torch.load(gen_path, map_location=device)
        g.load_state_dict(gen_state['model'])
        
        num_stage = gen_state['num_stage']
        
        if args.is_train and not args.new_optim:
            optim_g.load_state_dict(gen_state['optimizer'])
        
            # Move optimizer states to correct device
            for state in optim_g.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
    else:
        raise FileNotFoundError(f"No generator checkpoint found at {gen_path}")
    
    # Load discriminators
    for i, (disc, optim_d) in enumerate(zip(d_lst, optim_d_lst)):
        dis_path = os.path.join(checkpoint_path, f"epoch_{epoch}_Dis_{i}.pt")
        if os.path.exists(dis_path):
            if args.is_train:
                dis_state = torch.load(dis_path, map_location=device)
                disc.load_state_dict(dis_state['model'])
            
            
            if args.is_train and not args.new_optim:
                optim_d.load_state_dict(dis_state['optimizer'])
            
                # Move optimizer states to correct device
                for state in optim_d.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
        else:
            raise FileNotFoundError(f"No discriminator checkpoint found at {dis_path}")
    
    print(f'Loaded models from {checkpoint_path}')
    return epoch, num_stage

def save_metrics_to_csv(args, metrics):
    csv_path = os.path.join(args.result_path, 'metrics.csv')
    
    # Check if file exists to write headers
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write headers if file is new
        if not file_exists:
            writer.writerow(['epoch', 'clip_score', 
                           'fid_score', 'samples_processed'])
        
        # Write metrics
        writer.writerow([
            args.load_epoch,
            metrics['clip_score'],
            metrics['fid_score'],
            metrics['samples_processed']
        ])