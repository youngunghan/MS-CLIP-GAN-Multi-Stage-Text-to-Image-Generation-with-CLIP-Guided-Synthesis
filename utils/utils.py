import os
import sys
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import warnings
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


CHECKPOINT_FORMAT_VERSION = 2
LEGACY_CONDITIONING_ACTIVATION = 'relu'
LEGACY_ALIGNMENT_MODE = 'legacy_conditioned'


_TRAINING_SOURCE_PATTERNS = (
    'config/config.py',
    'criteria/diffaugment.py',
    'criteria/loss.py',
    'dataset/dataloader.py',
    'networks/*.py',
    'options/base_options.py',
    'options/train_options.py',
    'scripts/train.py',
    'scripts/trainer.py',
    'utils/utils.py',
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_path(path):
    """Hash a file or a directory tree deterministically."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'training data path does not exist: {path}')
    if path.is_file():
        return {
            'path': str(path),
            'kind': 'file',
            'size_bytes': path.stat().st_size,
            'sha256': _sha256_file(path),
        }

    digest = hashlib.sha256()
    total_size = 0
    file_count = 0
    for file_path in sorted(item for item in path.rglob('*') if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        file_hash = _sha256_file(file_path)
        size = file_path.stat().st_size
        digest.update(relative.encode('utf-8'))
        digest.update(b'\0')
        digest.update(file_hash.encode('ascii'))
        digest.update(b'\0')
        total_size += size
        file_count += 1
    return {
        'path': str(path),
        'kind': 'directory',
        'file_count': file_count,
        'size_bytes': total_size,
        'sha256': digest.hexdigest(),
    }


def _training_source_fingerprint(project_root):
    project_root = Path(project_root).resolve()
    paths = set()
    for pattern in _TRAINING_SOURCE_PATTERNS:
        paths.update(project_root.glob(pattern))
    files = {}
    aggregate = hashlib.sha256()
    for path in sorted(item for item in paths if item.is_file()):
        relative = path.relative_to(project_root).as_posix()
        file_hash = _sha256_file(path)
        files[relative] = file_hash
        aggregate.update(relative.encode('utf-8'))
        aggregate.update(b'\0')
        aggregate.update(file_hash.encode('ascii'))
        aggregate.update(b'\0')
    if not files:
        raise ValueError(f'no training source files found under {project_root}')
    return {'sha256': aggregate.hexdigest(), 'files': files}


def _git_training_state(project_root):
    def run_git(*arguments):
        try:
            result = subprocess.run(
                ['git'] + list(arguments), cwd=str(project_root),
                check=True, capture_output=True, text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run_git('rev-parse', 'HEAD')
    status = run_git('status', '--porcelain', '--untracked-files=no')
    return {
        'commit': commit,
        'dirty': None if status is None else bool(status),
    }


def _package_version(name):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _package_install_record(*names):
    """Return version plus PEP 610 VCS/direct-URL metadata when available."""
    for name in names:
        try:
            distribution = importlib_metadata.distribution(name)
        except importlib_metadata.PackageNotFoundError:
            continue
        direct_url = distribution.read_text('direct_url.json')
        try:
            direct_url = json.loads(direct_url) if direct_url else None
        except json.JSONDecodeError:
            direct_url = {'raw': direct_url}
        return {
            'distribution': name,
            'version': distribution.version,
            'direct_url': direct_url,
        }
    return None


def _python_package_source_fingerprint(name):
    """Hash installed Python source, distinguishing unversioned Git installs."""
    spec = importlib_util.find_spec(name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        roots = [Path(location).resolve() for location in spec.submodule_search_locations]
    elif spec.origin:
        roots = [Path(spec.origin).resolve().parent]
    else:
        return None

    files = {}
    aggregate = hashlib.sha256()
    for root_index, root in enumerate(roots):
        for path in sorted(root.rglob('*.py')):
            relative = f'{root_index}:{path.relative_to(root).as_posix()}'
            file_hash = _sha256_file(path)
            files[relative] = file_hash
            aggregate.update(relative.encode('utf-8'))
            aggregate.update(b'\0')
            aggregate.update(file_hash.encode('ascii'))
            aggregate.update(b'\0')
    if not files:
        return None
    return {'sha256': aggregate.hexdigest(), 'files': files}


def _nvidia_driver_versions():
    if not torch.cuda.is_available():
        return []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return sorted(set(line.strip() for line in result.stdout.splitlines() if line.strip()))


def build_training_provenance(data_path, device_ids=None, project_root=None):
    """Build stable provenance used to guard versioned exact resume.

    The dataset hash is over actual file bytes (or every file in a directory), and
    the source fingerprint is over all Python modules participating in this
    repository's training path. This can be expensive for a large dataset, so the
    training entry point computes it once and reuses it for every checkpoint.
    """
    root = (
        Path(project_root).resolve()
        if project_root is not None else Path(__file__).resolve().parents[1]
    )
    selected_ids = list(device_ids or [])
    gpu_properties = []
    if torch.cuda.is_available():
        for device_id in selected_ids:
            properties = torch.cuda.get_device_properties(device_id)
            gpu_properties.append({
                'device_id': int(device_id),
                'name': properties.name,
                'capability': [int(properties.major), int(properties.minor)],
                'total_memory': int(properties.total_memory),
            })

    return {
        'schema_version': 1,
        'dataset': _fingerprint_path(data_path),
        'source': _training_source_fingerprint(root),
        'git': _git_training_state(root),
        'runtime': {
            'python': platform.python_version(),
            'torch': str(torch.__version__),
            'torchvision': _package_version('torchvision'),
            'pillow': _package_version('Pillow'),
            'numpy': np.__version__,
            'clip_install': _package_install_record('clip', 'openai-clip'),
            'clip_source': _python_package_source_fingerprint('clip'),
            'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
        },
        'hardware': {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'cuda_available': torch.cuda.is_available(),
            'nvidia_driver': _nvidia_driver_versions(),
            'selected_gpus': gpu_properties,
        },
    }

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
    except TypeError:
        # PyTorch versions predating weights_only.
        return torch.load(path, map_location=device)
    except Exception:
        try:
            # Explicit False is required on PyTorch 2.6+, whose default changed.
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

def _move_optim_state_to_device(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _capture_rng_state():
    """Capture every RNG used by the training pipeline without advancing it."""
    numpy_state = np.random.get_state()
    state = {
        'python': random.getstate(),
        'numpy': {
            'bit_generator': numpy_state[0],
            # A Tensor keeps the checkpoint compatible with weights_only loading.
            # Older supported PyTorch versions cannot serialize torch.uint32.
            'keys': torch.from_numpy(numpy_state[1].astype(np.int64)),
            'position': numpy_state[2],
            'has_gauss': numpy_state[3],
            'cached_gaussian': numpy_state[4],
        },
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def _restore_rng_state(state):
    """Restore an RNG snapshot saved by :func:`_capture_rng_state`."""
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((
        numpy_state['bit_generator'],
        numpy_state['keys'].cpu().numpy().astype(np.uint32),
        int(numpy_state['position']),
        int(numpy_state['has_gauss']),
        float(numpy_state['cached_gaussian']),
    ))
    torch.set_rng_state(state['torch_cpu'].cpu())

    # Checkpoints may move between machines with different GPU counts. Restore all
    # matching devices; exact continuation naturally requires the same topology.
    if torch.cuda.is_available():
        for device_index, cuda_state in enumerate(state.get('torch_cuda', [])):
            if device_index >= torch.cuda.device_count():
                break
            torch.cuda.set_rng_state(cuda_state.cpu(), device=device_index)


def scheduler_horizon(num_epochs, resume_epoch=-1, new_optim=False,
                      schedule_config=None):
    """Return cosine T_max for a fresh, new-phase, or exact resumed run.

    A resumed v2 phase reuses its saved ``t_max``. This matters when a prior
    ``--new_optim`` extension started at (for example) epoch 150 with a 50-epoch
    cosine phase ending at epoch 200: a later interruption must recreate T_max=50,
    not T_max=200, before loading the scheduler state.
    """
    if not new_optim and resume_epoch >= 0 and schedule_config is not None:
        saved_end = schedule_config.get('phase_end_epoch')
        if saved_end is not None and int(saved_end) != int(num_epochs):
            raise ValueError(
                f'checkpoint scheduler phase ends at epoch {saved_end}, but '
                f'--num_epochs={num_epochs}; use the original phase end for exact '
                'resume or --new_optim to begin a new phase'
            )
        horizon = int(schedule_config['t_max'])
        if horizon <= 0:
            raise ValueError(f'invalid checkpoint scheduler t_max={horizon}')
        return horizon

    horizon = num_epochs
    if new_optim and resume_epoch >= 0:
        horizon = num_epochs - resume_epoch - 1
    if horizon <= 0:
        raise ValueError(
            f'no training epochs remain (num_epochs={num_epochs}, '
            f'resume_epoch={resume_epoch})'
        )
    return horizon


def _infer_legacy_model_config(checkpoint_state):
    """Infer the generator architecture encoded by historical parameter shapes."""
    model_state = _strip_module_prefix(checkpoint_state.get('model', {}))

    config = {
        'g_in_chans': None,
        'g_out_chans': None,
        'noise_dim': None,
        'condition_dim': None,
        'clip_embedding_dim': None,
        'num_stage': checkpoint_state.get('num_stage'),
        'conditioning_activation': LEGACY_CONDITIONING_ACTIVATION,
        'alignment_mode': LEGACY_ALIGNMENT_MODE,
    }

    cond_weight = model_state.get('cond_aug.layer.0.weight')
    if torch.is_tensor(cond_weight) and cond_weight.ndim == 2:
        config['condition_dim'] = int(cond_weight.shape[0] // 2)
        config['clip_embedding_dim'] = int(cond_weight.shape[1])

    mapping_first = model_state.get('g_layer.0.mapping_net.0.0.weight')
    if (torch.is_tensor(mapping_first) and mapping_first.ndim == 2
            and config['condition_dim'] is not None):
        config['noise_dim'] = int(mapping_first.shape[1] - config['condition_dim'])

    mapping_last = model_state.get('g_layer.0.mapping_net.7.0.weight')
    if (torch.is_tensor(mapping_last) and mapping_last.ndim == 2
            and mapping_last.shape[0] % 16 == 0):
        config['g_in_chans'] = int(mapping_last.shape[0] // 16)

    image_weight = model_state.get('g_layer.0.image_net.image_net.0.weight')
    if torch.is_tensor(image_weight) and image_weight.ndim == 4:
        # ConvTranspose2d stores [in_channels, out_channels, kH, kW].
        config['g_out_chans'] = int(image_weight.shape[1])

    return config


def _canonical_checkpoint_metadata(checkpoint_state):
    format_version = checkpoint_state.get('format_version')
    legacy = format_version is None
    if legacy:
        format_version = 1
        model_config = _infer_legacy_model_config(checkpoint_state)
        generator_weight_kind = 'legacy_unknown'
        scheduler_state = checkpoint_state.get('scheduler') or {}
        legacy_t_max = scheduler_state.get('T_max')
        schedule_config = None
        if legacy_t_max is not None:
            schedule_config = {
                'phase_start_epoch': 0,
                'phase_end_epoch': int(legacy_t_max),
                't_max': int(legacy_t_max),
                'scheduler_type': 'CosineAnnealingLR',
                'legacy_inferred': True,
            }
        training_config = None
        training_provenance = None
    else:
        if format_version > CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f'checkpoint format {format_version} is newer than supported '
                f'format {CHECKPOINT_FORMAT_VERSION}'
            )
        model_config = dict(checkpoint_state.get('model_config') or {})
        required = {
            'g_in_chans', 'g_out_chans', 'noise_dim', 'condition_dim',
            'clip_embedding_dim', 'num_stage', 'conditioning_activation',
            'alignment_mode'
        }
        missing = sorted(required.difference(model_config))
        if missing:
            raise ValueError(
                f'checkpoint format {format_version} is missing model_config fields: '
                f'{", ".join(missing)}'
            )
        generator_weight_kind = checkpoint_state.get('generator_weight_kind', 'raw')
        training_config = checkpoint_state.get('training_config')
        training_provenance = checkpoint_state.get('training_provenance')
        schedule_config = checkpoint_state.get('schedule_config')

    return {
        'format_version': int(format_version),
        'legacy': legacy,
        'generator_weight_kind': generator_weight_kind,
        'epoch': checkpoint_state.get('epoch'),
        'model_config': model_config,
        'training_config': training_config,
        'training_provenance': training_provenance,
        'schedule_config': schedule_config,
        'rng_state_available': checkpoint_state.get('rng_state') is not None,
    }


def peek_checkpoint_metadata(checkpoint_path, epoch, device='cpu'):
    """Read canonical generator metadata without constructing a model.

    Metadata-less checkpoints are reported as format 1 and receive the historical
    ``relu``/``legacy_conditioned`` behavior. Generator dimensions are inferred
    from their parameter shapes where possible. The returned mapping also exposes
    recorded training/scheduler configuration, provenance, and RNG availability.
    The helper only reads the record; training startup separately re-hashes the
    current dataset/source/runtime/hardware and exact resume compares them.
    """
    gen_path = os.path.join(str(checkpoint_path), f"epoch_{epoch}_Gen.pt")
    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"No generator checkpoint found at {gen_path}")
    return _canonical_checkpoint_metadata(_safe_torch_load(gen_path, device))


def _checkpoint_model_config(args, g, d_lst, num_stage):
    generator = _unwrap(g)
    discriminator = next((disc for disc in d_lst if disc is not None), None)
    if discriminator is not None:
        alignment_mode = getattr(_unwrap(discriminator), 'alignment_mode', None)
    else:
        alignment_mode = getattr(args, 'alignment_mode', None)

    config = {
        'g_in_chans': int(generator.in_chans),
        'g_out_chans': int(generator.out_chans),
        'noise_dim': int(generator.noise_dim),
        'condition_dim': int(generator.cond_dim),
        'clip_embedding_dim': int(generator.c_txt_dim),
        'num_stage': int(num_stage),
        'conditioning_activation': generator.conditioning_activation,
        'alignment_mode': alignment_mode,
    }
    if config['alignment_mode'] not in {'image_only', 'legacy_conditioned'}:
        raise ValueError(f"invalid alignment mode in checkpoint config: {alignment_mode!r}")
    return config


_TRAINING_DEFAULTS = {
    'seed': 42,
    'use_uncond_loss': False,
    'use_contrastive_loss': False,
    'use_mixed_loss': False,
    'use_mismatched_condition': True,
    'use_diffaugment': False,
    'diffaugment_policy': 'color,translation,cutout',
    'd_update_every': 1,
    'real_label_smooth': 1.0,
    'ema_decay': 0.999,
    'batch_size': 1,
    'num_workers': 4,
    'save_freq': 1,
}


def _base_learning_rates(optimizer, scheduler=None):
    if optimizer is None:
        return None
    if scheduler is not None and hasattr(scheduler, 'base_lrs'):
        return [float(value) for value in scheduler.base_lrs]
    return [
        float(group.get('initial_lr', group['lr']))
        for group in optimizer.param_groups
    ]


def _runtime_training_config(args, optim_g, optim_d_lst, scheduler_g,
                             scheduler_d_lst, use_ema):
    """Capture resume-relevant knobs visible to this training process.

    This mapping covers semantic knobs and base learning rates. Dataset/source/
    package/driver/hardware identity is recorded separately in
    ``training_provenance`` and compared alongside it.
    """
    config = {
        key: getattr(args, key, default)
        for key, default in _TRAINING_DEFAULTS.items()
    }
    config['use_ema'] = bool(use_ema)
    config['target_num_epochs'] = int(args.num_epochs)
    config['diffaugment_policy'] = ','.join(
        value.strip() for value in str(config['diffaugment_policy']).split(',')
        if value.strip()
    )
    config['g_learning_rates'] = _base_learning_rates(optim_g, scheduler_g)
    config['d_learning_rates'] = [
        _base_learning_rates(
            optimizer,
            scheduler_d_lst[index] if scheduler_d_lst is not None else None,
        )
        for index, optimizer in enumerate(optim_d_lst)
    ]
    return config


def _checkpoint_schedule_config(args, scheduler_g, scheduler_d_lst, epoch):
    if not hasattr(scheduler_g, 'T_max'):
        raise ValueError('checkpointing requires a cosine scheduler with T_max')
    t_max = int(scheduler_g.T_max)
    scheduler_type = scheduler_g.__class__.__name__
    for index, scheduler in enumerate(scheduler_d_lst):
        if (not hasattr(scheduler, 'T_max') or int(scheduler.T_max) != t_max
                or scheduler.__class__.__name__ != scheduler_type):
            raise ValueError(
                f'discriminator {index} scheduler T_max does not match generator'
            )
    phase_end_epoch = int(args.num_epochs)
    phase_start_epoch = phase_end_epoch - t_max
    if phase_start_epoch < 0 or not (phase_start_epoch <= epoch < phase_end_epoch):
        raise ValueError(
            f'invalid scheduler phase [{phase_start_epoch}, {phase_end_epoch}) '
            f'for checkpoint epoch {epoch} and T_max={t_max}'
        )
    expected_last_epoch = epoch - phase_start_epoch + 1
    if int(scheduler_g.last_epoch) != expected_last_epoch:
        raise ValueError(
            f'generator scheduler last_epoch={scheduler_g.last_epoch} is '
            f'inconsistent with checkpoint epoch {epoch} and phase start '
            f'{phase_start_epoch}; expected {expected_last_epoch}'
        )
    for index, scheduler in enumerate(scheduler_d_lst):
        if int(scheduler.last_epoch) != expected_last_epoch:
            raise ValueError(
                f'discriminator {index} scheduler last_epoch={scheduler.last_epoch} '
                f'is inconsistent with expected {expected_last_epoch}'
            )
    return {
        'phase_start_epoch': phase_start_epoch,
        # Exclusive, matching Python's range(..., num_epochs).
        'phase_end_epoch': phase_end_epoch,
        't_max': t_max,
        'scheduler_type': scheduler_type,
    }


def _config_values_equal(saved, current):
    if isinstance(saved, (list, tuple)) and isinstance(current, (list, tuple)):
        return len(saved) == len(current) and all(
            _config_values_equal(left, right)
            for left, right in zip(saved, current)
        )
    if (isinstance(saved, (int, float)) and not isinstance(saved, bool)
            and isinstance(current, (int, float)) and not isinstance(current, bool)):
        return math.isclose(float(saved), float(current), rel_tol=1e-12, abs_tol=1e-15)
    return saved == current


def _validate_training_config(args, metadata, optim_g, optim_d_lst,
                              scheduler_g, scheduler_d_lst, g_ema):
    """Fail before mutation when a v2 exact-resume contract has changed."""
    if not getattr(args, 'is_train', False) or getattr(args, 'new_optim', False):
        return
    if metadata['legacy']:
        warnings.warn(
            'metadata-less legacy checkpoint: optimizer/scheduler state will be '
            'restored where available, but training_config and RNG provenance are '
            'missing, so this is a best-effort legacy resume rather than a verified '
            'exact resume',
            RuntimeWarning,
        )
        return

    saved = metadata.get('training_config')
    if not isinstance(saved, dict):
        raise ValueError(
            'versioned checkpoint has no training_config; use --new_optim to '
            'start an intentional new training phase'
        )

    args_use_ema = bool(getattr(args, 'use_ema', g_ema is not None))
    if args_use_ema != (g_ema is not None):
        raise ValueError(
            f'current --use_ema={args_use_ema} but EMA model object presence is '
            f'{g_ema is not None}'
        )
    current = _runtime_training_config(
        args, optim_g, optim_d_lst, scheduler_g, scheduler_d_lst,
        use_ema=args_use_ema,
    )

    required = set(_TRAINING_DEFAULTS) | {
        'use_ema', 'target_num_epochs', 'g_learning_rates', 'd_learning_rates'
    }
    missing = sorted(required.difference(saved))
    if missing:
        raise ValueError(
            'checkpoint training_config is missing: ' + ', '.join(missing)
        )

    compare_fields = sorted(required)
    # Disabled features make their parameter semantically inactive.
    if not bool(saved['use_diffaugment']) and not bool(current['use_diffaugment']):
        compare_fields.remove('diffaugment_policy')
    if not bool(saved['use_ema']) and not bool(current['use_ema']):
        compare_fields.remove('ema_decay')

    mismatches = []
    for field in compare_fields:
        if not _config_values_equal(saved[field], current[field]):
            mismatches.append(
                f'{field}: checkpoint={saved[field]!r}, current={current[field]!r}'
            )
    if mismatches:
        raise ValueError(
            'exact resume training_config mismatch: ' + '; '.join(mismatches)
            + '. Use the saved settings or --new_optim for a new phase.'
        )


def _validate_training_provenance(args, metadata):
    if (not getattr(args, 'is_train', False)
            or getattr(args, 'new_optim', False)
            or metadata['legacy']):
        return
    saved = metadata.get('training_provenance')
    current = getattr(args, 'training_provenance', None)
    if not isinstance(saved, dict):
        raise ValueError(
            'versioned checkpoint has no training_provenance; use --new_optim '
            'for a new phase'
        )
    if not isinstance(current, dict):
        raise ValueError(
            'current training_provenance was not computed before exact resume'
        )

    comparisons = {
        'dataset.sha256': (
            saved.get('dataset', {}).get('sha256'),
            current.get('dataset', {}).get('sha256'),
        ),
        'dataset.size_bytes': (
            saved.get('dataset', {}).get('size_bytes'),
            current.get('dataset', {}).get('size_bytes'),
        ),
        'source.sha256': (
            saved.get('source', {}).get('sha256'),
            current.get('source', {}).get('sha256'),
        ),
        'runtime': (saved.get('runtime'), current.get('runtime')),
        'hardware': (saved.get('hardware'), current.get('hardware')),
    }
    mismatches = [
        f'{field}: checkpoint={left!r}, current={right!r}'
        for field, (left, right) in comparisons.items()
        if left is None or right is None or left != right
    ]
    if mismatches:
        raise ValueError(
            'exact resume training_provenance mismatch: ' + '; '.join(mismatches)
            + '. Use matching data/source/runtime/hardware or --new_optim.'
        )


def _set_checkpoint_compatibility_modes(g, d_lst, model_config, g_ema=None):
    conditioning_activation = model_config['conditioning_activation']
    alignment_mode = model_config['alignment_mode']

    for generator in (g, g_ema):
        if generator is not None:
            module = _unwrap(generator)
            if not hasattr(module, 'set_conditioning_activation'):
                raise TypeError('generator does not support conditioning compatibility modes')
            module.set_conditioning_activation(conditioning_activation)

    for discriminator in d_lst:
        if discriminator is not None:
            module = _unwrap(discriminator)
            if not hasattr(module, 'set_alignment_mode'):
                raise TypeError('discriminator does not support alignment compatibility modes')
            module.set_alignment_mode(alignment_mode)


def _validate_resume_scheduler(args, generator_state, metadata, scheduler_g=None):
    if not getattr(args, 'is_train', False) or getattr(args, 'new_optim', False):
        return
    if not hasattr(args, 'num_epochs'):
        raise ValueError('training resume requires args.num_epochs')
    scheduler_state = generator_state.get('scheduler')
    if scheduler_state is None:
        if metadata['legacy']:
            raise ValueError(
                'legacy checkpoint has no scheduler state, so its LR phase cannot '
                'be continued safely; use --new_optim to start an explicit new phase'
            )
        raise ValueError(
            'versioned checkpoint has no generator scheduler state; use '
            '--new_optim for a weight-only resume'
        )
    saved_horizon = int(scheduler_state.get('T_max'))

    if metadata['legacy']:
        if saved_horizon != args.num_epochs:
            raise ValueError(
                f'legacy checkpoint scheduler T_max={saved_horizon} does not '
                f'match --num_epochs={args.num_epochs}; use the historical horizon '
                'or --new_optim for a new phase'
            )
    else:
        schedule_config = metadata.get('schedule_config')
        if not isinstance(schedule_config, dict):
            raise ValueError(
                'versioned checkpoint has no schedule_config; use --new_optim '
                'for a new phase'
            )
        phase_end = int(schedule_config['phase_end_epoch'])
        phase_start = int(schedule_config['phase_start_epoch'])
        configured_t_max = int(schedule_config['t_max'])
        scheduler_type = schedule_config.get('scheduler_type')
        if phase_end != int(args.num_epochs):
            raise ValueError(
                f'checkpoint scheduler phase_end_epoch={phase_end} does not match '
                f'--num_epochs={args.num_epochs}'
            )
        if configured_t_max != saved_horizon:
            raise ValueError(
                f'checkpoint schedule_config.t_max={configured_t_max} differs '
                f'from scheduler state T_max={saved_horizon}'
            )
        expected_last_epoch = int(metadata['epoch']) - phase_start + 1
        saved_last_epoch = int(scheduler_state.get('last_epoch', -1))
        if saved_last_epoch != expected_last_epoch:
            raise ValueError(
                f'checkpoint scheduler state last_epoch={saved_last_epoch} is '
                f'inconsistent with expected {expected_last_epoch}'
            )
        if scheduler_g is None or int(getattr(scheduler_g, 'T_max', -1)) != saved_horizon:
            raise ValueError(
                f'current generator scheduler must be constructed with '
                f'T_max={saved_horizon} before exact resume'
            )
        if scheduler_type != scheduler_g.__class__.__name__:
            raise ValueError(
                f'checkpoint scheduler type={scheduler_type!r}, but current '
                f'scheduler type={scheduler_g.__class__.__name__!r}'
            )

def save_checkpoint(args, g: torch.nn.Module, d_lst: List[torch.nn.Module],
                optim_g, optim_d_lst, epoch: int, num_stage: int,
                scheduler_g=None, scheduler_d_lst=None, g_raw=None) -> None:
    """Save a versioned, exactly resumable training checkpoint.

    EMA runs pass EMA weights as ``g`` and live weights as ``g_raw``. A marker in
    the primary checkpoint makes the companion mandatory on resume, preventing an
    EMA model from ever being paired silently with a raw optimizer state.
    """
    if optim_g is None or scheduler_g is None:
        raise ValueError('training checkpoints require generator optimizer and scheduler')
    if len(d_lst) != len(optim_d_lst):
        raise ValueError('discriminator and optimizer list lengths differ')
    if (scheduler_d_lst is None or len(d_lst) != len(scheduler_d_lst)
            or any(scheduler is None for scheduler in scheduler_d_lst)):
        raise ValueError('training checkpoints require one scheduler per discriminator')
    actual_use_ema = g_raw is not None
    if hasattr(args, 'use_ema') and bool(args.use_ema) != actual_use_ema:
        raise ValueError(
            f'--use_ema={bool(args.use_ema)} does not match checkpoint payload '
            f'(EMA primary={actual_use_ema})'
        )
    model_config = _checkpoint_model_config(args, g, d_lst, num_stage)
    training_config = _runtime_training_config(
        args, optim_g, optim_d_lst, scheduler_g, scheduler_d_lst,
        use_ema=actual_use_ema,
    )
    schedule_config = _checkpoint_schedule_config(
        args, scheduler_g, scheduler_d_lst, epoch
    )
    training_provenance = getattr(args, 'training_provenance', None)
    if not isinstance(training_provenance, dict):
        raise ValueError(
            'training_provenance must be computed once at startup before saving '
            'a versioned checkpoint'
        )
    generator_weight_kind = 'ema' if g_raw is not None else 'raw'
    generator_state = {
        'format_version': CHECKPOINT_FORMAT_VERSION,
        'model_config': model_config,
        'training_config': training_config,
        'training_provenance': training_provenance,
        'schedule_config': schedule_config,
        'generator_weight_kind': generator_weight_kind,
        'model': _unwrap(g).state_dict(),
        'optimizer': optim_g.state_dict(),
        'scheduler': scheduler_g.state_dict() if scheduler_g is not None else None,
        'epoch': epoch,
        'num_stage': num_stage,
        'rng_state': _capture_rng_state(),
    }
    torch.save(generator_state,
              os.path.join(args.checkpoint_path, f"epoch_{epoch}_Gen.pt"))

    if g_raw is not None:
        raw_state = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'model_config': model_config,
            'training_config': training_config,
            'training_provenance': training_provenance,
            'schedule_config': schedule_config,
            'generator_weight_kind': 'raw_companion',
            'model': _unwrap(g_raw).state_dict(),
            'epoch': epoch,
            'num_stage': num_stage,
        }
        torch.save(
            raw_state,
            os.path.join(args.checkpoint_path, f"epoch_{epoch}_Gen_raw.pt")
        )

    # Save discriminators
    for i, (disc, optim_d) in enumerate(zip(d_lst, optim_d_lst)):
        sched_d = scheduler_d_lst[i] if scheduler_d_lst is not None else None
        discriminator_state = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'model_config': model_config,
            'training_config': training_config,
            'training_provenance': training_provenance,
            'schedule_config': schedule_config,
            'discriminator_index': i,
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
    """Load models and enforce the versioned resume contract before mutation.

    - Handles checkpoints saved with or without a 'module.' prefix.
    - Restores optimizer, LR scheduler, compatibility modes, and all recorded RNGs.
    - Versioned exact resume compares loss/stability/data-loader/checkpoint cadence,
      EMA, base learning rates, and scheduler-phase settings.
    - During inference, d_lst entries are None and discriminator files are not required.
    - EMA checkpoints store the EMA weights in Gen.pt and the live training weights in
      Gen_raw.pt: when training, the raw weights go into `g` and Gen.pt seeds `g_ema`.

    "Exact" verifies recorded dataset/source/package/driver/hardware fingerprints
    and RNG/configuration state. It still cannot guarantee deterministic behavior
    from every CUDA kernel or detect external inputs outside the recorded paths.
    """
    if len(d_lst) != len(optim_d_lst):
        raise ValueError('discriminator and optimizer list lengths differ')
    device = next(g.parameters()).device

    # Load generator
    gen_path = os.path.join(checkpoint_path, f"epoch_{epoch}_Gen.pt")
    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"No generator checkpoint found at {gen_path}")

    gen_state = _safe_torch_load(gen_path, device)
    metadata = _canonical_checkpoint_metadata(gen_state)
    if metadata['epoch'] is not None and metadata['epoch'] != epoch:
        raise ValueError(
            f'generator checkpoint requested as epoch {epoch} contains epoch '
            f'{metadata["epoch"]}'
        )
    _validate_resume_scheduler(args, gen_state, metadata, scheduler_g=scheduler_g)
    exact_versioned_resume = (
        args.is_train and not args.new_optim and not metadata['legacy']
    )
    if args.is_train and not args.new_optim:
        if not metadata['legacy'] and scheduler_g is None:
            raise ValueError(
                'exact training resume requires a generator scheduler object; '
                'use --new_optim for a weight-only resume'
            )
        if (
            not metadata['legacy']
            and (
                scheduler_d_lst is None
                or len(scheduler_d_lst) != len(d_lst)
                or any(scheduler is None for scheduler in scheduler_d_lst)
            )
        ):
            raise ValueError(
                'exact training resume requires one scheduler object per '
                'discriminator; use --new_optim for a weight-only resume'
            )
    _validate_training_provenance(args, metadata)
    _validate_training_config(
        args, metadata, optim_g, optim_d_lst, scheduler_g,
        scheduler_d_lst, g_ema,
    )

    weight_kind = metadata['generator_weight_kind']
    if exact_versioned_resume:
        requested_ema = bool(getattr(args, 'use_ema', False))
        if weight_kind == 'ema' and (not requested_ema or g_ema is None):
            raise ValueError(
                'checkpoint primary contains EMA weights; exact resume requires '
                '--use_ema and a live EMA model object'
            )
        if weight_kind == 'raw' and (requested_ema or g_ema is not None):
            raise ValueError(
                'checkpoint primary contains raw weights; exact resume cannot '
                'enable EMA. Use --new_optim to start a new EMA phase.'
            )
    _set_checkpoint_compatibility_modes(
        g, d_lst, metadata['model_config'], g_ema=g_ema
    )

    raw_path = os.path.join(checkpoint_path, f"epoch_{epoch}_Gen_raw.pt")
    raw_exists = os.path.exists(raw_path)
    if exact_versioned_resume and weight_kind == 'ema' and not raw_exists:
        raise FileNotFoundError(
            f'{gen_path} is marked as EMA weights but its required raw companion '
            f'is missing: {raw_path}'
        )
    if exact_versioned_resume and weight_kind == 'raw' and raw_exists:
        raise ValueError(
            f'{gen_path} is marked as raw weights but an unexpected raw companion '
            f'exists at {raw_path}; refusing an ambiguous resume'
        )
    if not metadata['legacy'] and weight_kind not in {'raw', 'ema'}:
        raise ValueError(f'unsupported generator_weight_kind={weight_kind!r}')

    use_raw_companion = args.is_train and raw_exists and (
        metadata['legacy'] or weight_kind == 'ema'
    )
    if use_raw_companion:
        raw_state = _safe_torch_load(raw_path, device)
        if not metadata['legacy']:
            raw_metadata = _canonical_checkpoint_metadata(raw_state)
            if raw_metadata['generator_weight_kind'] != 'raw_companion':
                raise ValueError(
                    f'EMA raw companion has invalid generator_weight_kind: '
                    f'{raw_metadata["generator_weight_kind"]!r}'
                )
            if raw_metadata['epoch'] != epoch:
                raise ValueError(
                    f'EMA raw companion epoch {raw_metadata["epoch"]} does not '
                    f'match requested epoch {epoch}'
                )
            if raw_metadata['model_config'] != metadata['model_config']:
                raise ValueError('EMA and raw companion model_config values differ')
            if raw_metadata['training_config'] != metadata['training_config']:
                raise ValueError('EMA and raw companion training_config values differ')
            if raw_metadata['training_provenance'] != metadata['training_provenance']:
                raise ValueError('EMA and raw companion training_provenance values differ')
            if raw_metadata['schedule_config'] != metadata['schedule_config']:
                raise ValueError('EMA and raw companion schedule_config values differ')
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
        if exact_versioned_resume and (
                optim_g is None or gen_state.get('optimizer') is None):
            raise ValueError(
                'exact training resume requires generator optimizer state; '
                'use --new_optim for a weight-only resume'
            )
        if optim_g is not None and gen_state.get('optimizer') is not None:
            optim_g.load_state_dict(gen_state['optimizer'])
            _move_optim_state_to_device(optim_g, device)
        elif metadata['legacy']:
            warnings.warn(
                'legacy checkpoint has no restorable generator optimizer state',
                RuntimeWarning,
            )
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
        if dis_state.get('epoch') is not None and dis_state['epoch'] != epoch:
            raise ValueError(
                f'discriminator {i} checkpoint contains epoch {dis_state["epoch"]}, '
                f'expected {epoch}'
            )
        if not metadata['legacy']:
            dis_config = dis_state.get('model_config')
            if dis_config != metadata['model_config']:
                raise ValueError(
                    f'discriminator {i} model_config does not match generator checkpoint'
                )
            if dis_state.get('training_config') != metadata['training_config']:
                raise ValueError(
                    f'discriminator {i} training_config does not match generator checkpoint'
                )
            if dis_state.get('training_provenance') != metadata['training_provenance']:
                raise ValueError(
                    f'discriminator {i} training_provenance does not match '
                    'generator checkpoint'
                )
            if dis_state.get('schedule_config') != metadata['schedule_config']:
                raise ValueError(
                    f'discriminator {i} schedule_config does not match generator checkpoint'
                )
        if args.is_train and not args.new_optim:
            dis_scheduler_state = dis_state.get('scheduler')
            if dis_scheduler_state is None and exact_versioned_resume:
                raise ValueError(
                    f'exact training resume requires discriminator {i} scheduler '
                    'state; use --new_optim for a weight-only resume'
                )
            if dis_scheduler_state is not None:
                expected_t_max = (
                    int(metadata['schedule_config']['t_max'])
                    if not metadata['legacy'] else int(args.num_epochs)
                )
                if int(dis_scheduler_state.get('T_max')) != expected_t_max:
                    raise ValueError(
                        f'discriminator {i} scheduler T_max='
                        f'{dis_scheduler_state.get("T_max")} does not match '
                        f'expected phase T_max={expected_t_max}'
                    )
                if not metadata['legacy']:
                    expected_last_epoch = (
                        int(metadata['epoch'])
                        - int(metadata['schedule_config']['phase_start_epoch'])
                        + 1
                    )
                    if int(dis_scheduler_state.get('last_epoch', -1)) != expected_last_epoch:
                        raise ValueError(
                            f'discriminator {i} scheduler state last_epoch='
                            f'{dis_scheduler_state.get("last_epoch")} does not '
                            f'match expected {expected_last_epoch}'
                        )
                sched_d = (
                    scheduler_d_lst[i]
                    if scheduler_d_lst is not None and i < len(scheduler_d_lst)
                    else None
                )
                if (exact_versioned_resume and
                        int(getattr(sched_d, 'T_max', -1)) != expected_t_max):
                    raise ValueError(
                        f'current discriminator {i} scheduler must be constructed '
                        f'with T_max={expected_t_max}'
                    )
                if (exact_versioned_resume and
                        sched_d.__class__.__name__ !=
                        metadata['schedule_config']['scheduler_type']):
                    raise ValueError(
                        f'current discriminator {i} scheduler type '
                        f'{sched_d.__class__.__name__!r} does not match '
                        f'{metadata["schedule_config"]["scheduler_type"]!r}'
                    )
            elif metadata['legacy']:
                warnings.warn(
                    f'legacy discriminator {i} has no scheduler state; retaining '
                    'the newly constructed scheduler',
                    RuntimeWarning,
                )
        _unwrap(disc).load_state_dict(_strip_module_prefix(dis_state['model']))

        if args.is_train and not args.new_optim:
            if exact_versioned_resume and (
                    optim_d is None or dis_state.get('optimizer') is None):
                raise ValueError(
                    f'exact training resume requires discriminator {i} optimizer '
                    'state; use --new_optim for a weight-only resume'
                )
            if optim_d is not None and dis_state.get('optimizer') is not None:
                optim_d.load_state_dict(dis_state['optimizer'])
                _move_optim_state_to_device(optim_d, device)
            elif metadata['legacy']:
                warnings.warn(
                    f'legacy checkpoint has no restorable discriminator {i} '
                    'optimizer state',
                    RuntimeWarning,
                )
            sched_d = scheduler_d_lst[i] if scheduler_d_lst is not None else None
            if sched_d is not None and dis_state.get('scheduler') is not None:
                sched_d.load_state_dict(dis_state['scheduler'])

    if args.is_train:
        # Unlike the optimizer/scheduler/EMA blocks above, RNG restore is NOT
        # gated on --new_optim: per the documented contract (train-eval-infer.md,
        # "v2의 RNG만 복구한다"), --new_optim intentionally still restores the RNG
        # state so the new cosine phase itself remains reproducible/re-resumable,
        # even though it drops optimizer/scheduler state.
        if gen_state.get('rng_state') is not None:
            _restore_rng_state(gen_state['rng_state'])
        else:
            warnings.warn(
                'legacy checkpoint has no RNG state; weights load correctly but the '
                'resumed stochastic sequence cannot be exact',
                RuntimeWarning
            )

    if metadata['legacy'] and args.is_train:
        compatibility = 'legacy best-effort resume'
    elif metadata['legacy']:
        compatibility = 'legacy compatibility'
    else:
        compatibility = 'format v2'
    print(
        f'Loaded models from {checkpoint_path} ({compatibility}; '
        f'conditioning={metadata["model_config"]["conditioning_activation"]}, '
        f'alignment={metadata["model_config"]["alignment_mode"]})'
    )
    return epoch, num_stage

def save_metrics_to_csv(args, metrics):
    csv_path = os.path.join(str(args.result_path), 'metrics.csv')
    fields = [
        'checkpoint_path', 'checkpoint_sha256', 'format_version', 'legacy',
        'generator_weight_kind', 'conditioning_activation', 'alignment_mode',
        'epoch', 'eval_seed', 'clip_score', 'fid_score',
        'inception_score_mean', 'inception_score_std', 'samples_processed',
    ]
    row = {
        'checkpoint_path': args.eval_checkpoint_path,
        'checkpoint_sha256': args.eval_checkpoint_sha256,
        'format_version': args.eval_checkpoint_format,
        'legacy': args.eval_checkpoint_legacy,
        'generator_weight_kind': args.eval_generator_weight_kind,
        'conditioning_activation': args.eval_conditioning_activation,
        'alignment_mode': args.eval_alignment_mode,
        'epoch': args.load_epoch,
        'eval_seed': args.seed,
        'clip_score': metrics['clip_score'],
        'fid_score': metrics['fid_score'],
        'inception_score_mean': metrics.get('inception_score_mean', ''),
        'inception_score_std': metrics.get('inception_score_std', ''),
        'samples_processed': metrics['samples_processed'],
    }

    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path, newline='') as existing:
            header = next(csv.reader(existing), None)
        if header != fields:
            raise ValueError(
                f'{csv_path} uses an older/incompatible schema; move or remove it '
                'before recording provenance-aware metrics'
            )

    with open(csv_path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
