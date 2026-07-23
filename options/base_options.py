import argparse
import os
import time
from pathlib import Path
from utils.utils import *


def training_run_suffix():
    """Return a collision-resistant run suffix without consuming any RNG state."""
    timestamp_ns = time.time_ns()
    timestamp = time.strftime(
        "-%Y_%m_%d_%H_%M_%S",
        time.localtime(timestamp_ns // 1_000_000_000),
    )
    return f'{timestamp}_{timestamp_ns % 1_000_000_000:09d}-p{os.getpid()}'

def str2bool(v):
    """argparse type for real booleans — plain type=bool would parse '--flag False' as True."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    if v.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f'boolean value expected, got {v!r}')


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--seed', type=int, default=42,
                            help='random seed for reproducibility')
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--num_workers', type=int, default=4)

        parser.add_argument('--data_path', default='./data/sample_train.zip', type=Path, help="path of directory containing training dataset")
        parser.add_argument('--resume_checkpoint_path', default=None)
        parser.add_argument('--resume_epoch', type=int, default=-1)
        parser.add_argument('--report_interval', type=int, default=100, help='Report interval')
        parser.add_argument('--checkpoint_path', type=Path, default='./checkpoints', help='Checkpoint path')
        parser.add_argument('--result_path', type=Path, default='./output', help='Generated image path')

        parser.add_argument('--noise_dim', type=int, default=100, help= 'Input noise dimension to Generator')
        parser.add_argument('--condition_dim', type=int, default=128, help= 'Noise projection dimension')
        parser.add_argument('--clip_embedding_dim', type=int, default=512, help='Dimension of c_txt from CLIP ViT-B/32 (FIXED at 512; validated in parse())')

        parser.add_argument('--g_in_chans', type=int, default=1024,
                            help='Number of input channels for generator (Ng)')
        parser.add_argument('--g_out_chans', type=int, default=3,
                            help='Number of output channels for generator')
        parser.add_argument('--d_in_chans', type=int, default=64,
                            help='Number of input channels for discriminator (Nd)')
        parser.add_argument('--d_out_chans', type=int, default=1,
                            help='Number of output channels for discriminator')
        parser.add_argument('--num_stage', type=int, default=3)
        parser.add_argument(
            '--conditioning_activation', choices=['linear', 'relu'], default='linear',
            help='Conditioning augmentation projection. Fresh runs default to linear '
                 'after saturation was measured in this repository\'s legacy checkpoint; '
                 'checkpoint metadata overrides this when loading.'
        )
        parser.add_argument(
            '--alignment_mode', choices=['image_only', 'legacy_conditioned'],
            default='image_only',
            help='Alignment-head input mode. image_only prevents the text shortcut; '
                 'checkpoint metadata overrides this when loading.'
        )
        parser.add_argument(
            '--deterministic_cond', action='store_true',
            help='Make conditioning augmentation deterministic: ConditioningAugmention '
                 'returns condition = mu (no reparameterization noise). Fresh runs '
                 'default to the original stochastic behaviour; checkpoint metadata '
                 'overrides this when loading.'
        )

        # Pipeline is fixed to CLIP ViT-B/32 (512-dim): preprocessing (preprocess_dataset.py)
        # HARDCODES ViT-B/32 when computing the stored embeddings, and --clip_embedding_dim is 512
        # (validated in parse()). Another model (e.g. ViT-L/14 = 768-dim) would mismatch the
        # generator/contrastive dims. Supporting one is a CODE change — edit the hardcoded model in
        # preprocessing/preprocess_dataset.py, re-preprocess, and match --clip_embedding_dim — not
        # just a CLI flag, so the choice is restricted here.
        parser.add_argument('--clip_model', type=str, choices=['ViT-B/32'], default='ViT-B/32')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the options (parse_args, not parse_known_args, so typo'd CLI flags error
        # loudly instead of being silently ignored). All sub-class options are already
        # registered via initialize(), so this is safe.
        opt = parser.parse_args()
        self.parser = parser

        return opt

    def validate(self, opt):
        """Reject architecture/CLI contracts that would fail or train nonsense."""
        def require(condition, message):
            if not condition:
                self.parser.error(message)

        require(opt.num_workers >= 0, '--num_workers must be >= 0')
        require(opt.report_interval > 0, '--report_interval must be > 0')
        require(opt.noise_dim > 0, '--noise_dim must be > 0')
        require(opt.condition_dim > 0, '--condition_dim must be > 0')
        require(opt.g_in_chans > 0, '--g_in_chans must be > 0')
        require(opt.d_in_chans > 0, '--d_in_chans must be > 0')
        require(opt.num_stage > 0, '--num_stage must be > 0')
        require(opt.g_out_chans == 3,
                '--g_out_chans must be 3 (RGB dataset, CLIP, and VGG contract)')
        require(opt.d_out_chans == 1,
                '--d_out_chans must be 1 (scalar BCE discriminator contract)')

        # Stage 0 downsamples channels by /16. Later stages must receive at least
        # four channels because their SSA channel-attention bottleneck uses C//4.
        require(
            opt.g_in_chans % 16 == 0,
            '--g_in_chans must be divisible by 16 for the four stage-0 upsamplers'
        )
        if opt.num_stage > 1:
            final_refinement_input = opt.g_in_chans // (16 * (2 ** (opt.num_stage - 2)))
            require(
                final_refinement_input >= 4,
                '--g_in_chans is too small for the requested --num_stage '
                '(the last refinement stage needs at least four channels)'
            )

        try:
            gpu_ids = [int(value) for value in opt.gpu_ids.split(',')]
        except ValueError:
            self.parser.error('--gpu_ids must be a comma-separated list of integers')
        require(len(gpu_ids) > 0, '--gpu_ids must not be empty')
        require(all(value >= -1 for value in gpu_ids),
                '--gpu_ids entries must be >= -1')
        require(-1 not in gpu_ids or gpu_ids == [-1],
                '--gpu_ids -1 (CPU) cannot be combined with CUDA ids')

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        return message

    def parse(self, print_options=True):
        opt = self.gather_options()

        # Pipeline is fixed to CLIP ViT-B/32 (512-dim); guard against a mismatched embedding dim
        # (preprocessing hardcodes ViT-B/32 → stored features are 512-d). See --clip_model.
        if opt.clip_embedding_dim != 512:
            error(f"--clip_embedding_dim must be 512 (CLIP ViT-B/32); got {opt.clip_embedding_dim}. "
                  f"The pipeline is fixed to ViT-B/32 — changing it requires editing preprocessing "
                  f"(preprocess_dataset.py) and re-preprocessing, not just this flag.")
        self.validate(opt)
        seed_fix(opt.seed)
        is_train = bool(getattr(opt, 'is_train', False))
        expr_dir = None
        if is_train:
            opt.name = opt.name + training_run_suffix()
            checkpoint_root = Path(opt.checkpoint_path)
            expr_dir = checkpoint_root / opt.name
            # Both default and custom checkpoint roots are namespaces, never the
            # checkpoint directory itself. This prevents independent runs from
            # overwriting epoch_<N> files in a shared custom root.
            opt.checkpoint_path = expr_dir / 'ckpt'

            # Preserve the established meaning of an explicit result path: it is
            # the final output directory. Only the default follows the run folder.
            default_result = Path(self.parser.get_default('result_path'))
            if Path(opt.result_path) == default_result:
                opt.result_path = expr_dir / 'res'

            if not print_options and expr_dir.exists():
                raise FileExistsError(
                    f'training run namespace already exists: {expr_dir}'
                )

        if print_options:
            msg = self.print_options(opt)
            if is_train:
                # Option snapshots and experiment directories are training-only.
                # Evaluation/inference checkpoint paths are strictly read-only.
                try:
                    # Atomic namespace claim: never attach a new process to an
                    # existing run, even if its name was generated concurrently.
                    expr_dir.mkdir(parents=True, exist_ok=False)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f'training run namespace already exists: {expr_dir}'
                    ) from exc
                mkdirs(opt.checkpoint_path)
                if Path(opt.result_path) == expr_dir / 'res':
                    mkdirs(opt.result_path)
                file_name = expr_dir / 'opt.txt'
                with open(file_name, 'wt') as opt_file:
                    opt_file.write(msg)
                    opt_file.write('\n')

        # parse gpu ids string into a list of ints.
        # NOTE: actual device selection is handled by the scripts (and CUDA_VISIBLE_DEVICES);
        # we intentionally do NOT call torch.cuda.set_device here to avoid mixing
        # physical ids with masked/visible indices.
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        self.opt = opt
        return self.opt
