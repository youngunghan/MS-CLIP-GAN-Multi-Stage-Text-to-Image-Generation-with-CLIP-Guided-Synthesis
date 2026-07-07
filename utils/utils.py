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

@torch.no_grad()
def ema_update(ema_model, model, decay):
    """In-place EMA: ema = decay*ema + (1-decay)*model (params), buffers copied.

    Both models are unwrapped from DataParallel first so parameter names line up.
    Buffers (e.g. BatchNorm running stats) are copied straight across so the EMA
    generator stays usable in eval mode.
    """
    ema, src = _unwrap(ema_model), _unwrap(model)
    ema_params = dict(ema.named_parameters())
    for name, p in src.named_parameters():
        ema_params[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    ema_bufs = dict(ema.named_buffers())
    for name, b in src.named_buffers():
        if name in ema_bufs:
            ema_bufs[name].copy_(b)

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

def _unwrap(model):
    """Return the underlying module if wrapped in DataParallel/DDP, else the model itself."""
    return model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model

def _strip_module_prefix(state_dict):
    """Drop a leading 'module.' from every key (left over from DataParallel saves)."""
    if state_dict and all(k.startswith('module.') for k in state_dict):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict

def _safe_torch_load(path, device):
    """Load a checkpoint, preferring weights_only=True.

    Falls back to a plain (full) load if weights_only is unsupported (older PyTorch)
    or fails to deserialize (e.g. optimizer/scheduler state with non-tensor objects).
    These are local, trusted checkpoints produced by this codebase.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device)

def _move_optim_state_to_device(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def save_checkpoint(args, g: torch.nn.Module, d_lst: List[torch.nn.Module],
                optim_g, optim_d_lst, epoch: int, num_stage: int,
                scheduler_g=None, scheduler_d_lst=None, g_raw=None) -> None:
    """Save generator and discriminator models (always unwrapping DataParallel).

    Optimizer AND LR-scheduler states are stored so that resume continues the
    learning-rate schedule from the correct epoch instead of restarting it.

    EMA runs pass the EMA model as `g` (so Gen.pt is what eval/infer should load)
    and the live training generator as `g_raw`; without the extra Gen_raw.pt file,
    resume would restart training from the time-averaged weights.
    """
    # Save generator
    generator_state = {
        'model': _unwrap(g).state_dict(),
        'optimizer': optim_g.state_dict(),
        'scheduler': scheduler_g.state_dict() if scheduler_g is not None else None,
        'epoch': epoch,
        'num_stage': num_stage
    }
    torch.save(generator_state,
              os.path.join(args.checkpoint_path, f"epoch_{epoch}_Gen.pt"))

    if g_raw is not None:
        torch.save({'model': _unwrap(g_raw).state_dict(), 'epoch': epoch, 'num_stage': num_stage},
                   os.path.join(args.checkpoint_path, f"epoch_{epoch}_Gen_raw.pt"))

    # Save discriminators
    for i, (disc, optim_d) in enumerate(zip(d_lst, optim_d_lst)):
        sched_d = scheduler_d_lst[i] if scheduler_d_lst is not None else None
        discriminator_state = {
            'model': _unwrap(disc).state_dict(),
            'optimizer': optim_d.state_dict(),
            'scheduler': sched_d.state_dict() if sched_d is not None else None,
            'epoch': epoch,
            'num_stage': num_stage
        }
        torch.save(discriminator_state,
                  os.path.join(args.checkpoint_path, f"epoch_{epoch}_Dis_{i}.pt"))

    print(f'Saved models to {args.checkpoint_path}')

def load_checkpoint(args, g: torch.nn.Module, d_lst: List[torch.nn.Module],
               optim_g, optim_d_lst: List[torch.optim.Optimizer], checkpoint_path, epoch: int,
               scheduler_g=None, scheduler_d_lst=None, g_ema=None) -> tuple:
    """Load generator and (optionally) discriminator models with optimizers/schedulers.

    - Handles checkpoints saved with or without a 'module.' prefix.
    - Restores optimizer AND LR-scheduler state when training (so resume continues the schedule).
    - During inference, d_lst entries are None and discriminator files are not required.
    - EMA checkpoints store the EMA weights in Gen.pt and the live training weights in
      Gen_raw.pt: when training, the raw weights go into `g` and Gen.pt seeds `g_ema`.
    """
    device = next(g.parameters()).device

    # Load generator
    gen_path = os.path.join(checkpoint_path, f"epoch_{epoch}_Gen.pt")
    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"No generator checkpoint found at {gen_path}")

    gen_state = _safe_torch_load(gen_path, device)
    raw_path = os.path.join(checkpoint_path, f"epoch_{epoch}_Gen_raw.pt")
    if args.is_train and os.path.exists(raw_path):
        raw_state = _safe_torch_load(raw_path, device)
        _unwrap(g).load_state_dict(_strip_module_prefix(raw_state['model']))
        if g_ema is not None:
            _unwrap(g_ema).load_state_dict(_strip_module_prefix(gen_state['model']))
        else:
            print('Note: EMA checkpoint resumed without --use_ema; training continues '
                  'from the raw weights and the saved EMA average is dropped.')
    else:
        _unwrap(g).load_state_dict(_strip_module_prefix(gen_state['model']))
        if g_ema is not None:
            # Non-EMA checkpoint (or one predating Gen_raw.pt): restart the average
            # from the loaded weights.
            _unwrap(g_ema).load_state_dict(_unwrap(g).state_dict())
    num_stage = gen_state['num_stage']

    if args.is_train and not args.new_optim:
        if optim_g is not None and gen_state.get('optimizer') is not None:
            optim_g.load_state_dict(gen_state['optimizer'])
            _move_optim_state_to_device(optim_g, device)
        if scheduler_g is not None and gen_state.get('scheduler') is not None:
            scheduler_g.load_state_dict(gen_state['scheduler'])

    # Load discriminators (skipped entirely for inference where disc is None)
    for i, (disc, optim_d) in enumerate(zip(d_lst, optim_d_lst)):
        if disc is None:
            continue

        dis_path = os.path.join(checkpoint_path, f"epoch_{epoch}_Dis_{i}.pt")
        if not os.path.exists(dis_path):
            raise FileNotFoundError(f"No discriminator checkpoint found at {dis_path}")

        dis_state = _safe_torch_load(dis_path, device)
        _unwrap(disc).load_state_dict(_strip_module_prefix(dis_state['model']))

        if args.is_train and not args.new_optim:
            if optim_d is not None and dis_state.get('optimizer') is not None:
                optim_d.load_state_dict(dis_state['optimizer'])
                _move_optim_state_to_device(optim_d, device)
            sched_d = scheduler_d_lst[i] if scheduler_d_lst is not None else None
            if sched_d is not None and dis_state.get('scheduler') is not None:
                sched_d.load_state_dict(dis_state['scheduler'])

    print(f'Loaded models from {checkpoint_path}')
    return epoch, num_stage

def save_metrics_to_csv(args, metrics):
    csv_path = os.path.join(str(args.result_path), 'metrics.csv')

    # Check if file exists to write headers
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write headers if file is new
        if not file_exists:
            writer.writerow(['epoch', 'clip_score', 'fid_score',
                             'inception_score_mean', 'inception_score_std',
                             'samples_processed'])

        # Write metrics
        writer.writerow([
            args.load_epoch,
            metrics['clip_score'],
            metrics['fid_score'],
            metrics.get('inception_score_mean', ''),
            metrics.get('inception_score_std', ''),
            metrics['samples_processed']
        ])