from .base_options import BaseOptions, str2bool

class TrainOptions(BaseOptions):
    def initialize(self, parser):  
        parser = BaseOptions.initialize(self, parser)     
        parser.add_argument('--batch_size', type=int, default=1, help='Batch size') #64
        parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
        parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')   
        parser.add_argument('--save_freq', type=int, default=1, help='Frequency of saving checkpoints (epochs)')
        parser.add_argument('--use_uncond_loss', action="store_true")
        parser.add_argument('--use_contrastive_loss', action="store_true")
        parser.add_argument('--use_mixed_loss', action="store_true")
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')
        parser.add_argument(
            '--no_mismatched_condition', action='store_false',
            dest='use_mismatched_condition',
            help='Disable the default real-image/mismatched-text conditional BCE negative.'
        )
        parser.set_defaults(use_mismatched_condition=True)

        # --- GAN stability levers (all opt-in; defaults reproduce the original behaviour) ---
        parser.add_argument('--d_lr', type=float, default=-1.0,
                            help='Discriminator LR (TTUR). -1 means use --learning_rate for D too.')
        parser.add_argument('--use_ema', action='store_true',
                            help='Track an EMA of the generator weights and sample/checkpoint the EMA model.')
        parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay for the generator.')
        parser.add_argument('--real_label_smooth', type=float, default=1.0,
                            help='Discriminator real-label target (e.g. 0.9 for one-sided label smoothing).')
        parser.add_argument('--d_update_every', type=int, default=1,
                            help='Update D once every N generator steps (N>1 weakens D, i.e. n_critic<1).')
        parser.add_argument('--use_diffaugment', action='store_true',
                            help='Apply DiffAugment (differentiable aug on BOTH real and fake) at the discriminator.')
        parser.add_argument('--diffaugment_policy', type=str, default='color,translation,cutout',
                            help='DiffAugment policy (comma-separated subset of color,translation,cutout).')

        # --- Text-conditioning pressure: tunable weights + staged introduction ---
        parser.add_argument('--gamma', type=float, default=5.0,
                            help='D-side text-image alignment InfoNCE weight (the two '
                                 'contrastive_loss_D terms in D_loss). Default 5 reproduces '
                                 'the original hardcoded weight.')
        parser.add_argument('--lam', type=float, default=10.0,
                            help='G-side CLIP contrastive weight, applied only at stages whose '
                                 'resolution is >= CLIPConfig.MIN_QUALITY_SIZE. Default 10 '
                                 'reproduces the original hardcoded weight.')
        parser.add_argument('--cond_warmup_epochs', type=int, default=0,
                            help='Number of initial epochs during which the D-side conditioning-'
                                 'pressure terms that share the discriminator image trunk (the '
                                 'gamma alignment InfoNCE terms and the mismatched-condition '
                                 'negative) are held off, so D first learns real/fake separation '
                                 'undisturbed. The plain real/fake BCE always trains from epoch 0. '
                                 '0 (default) reproduces the original always-on behaviour.')
        parser.add_argument('--cond_ramp_epochs', type=int, default=0,
                            help='Number of epochs, after --cond_warmup_epochs ends, over which '
                                 'the gated conditioning terms linearly ramp from 0 to full weight '
                                 'instead of switching on abruptly. 0 (default) is a hard switch.')

        parser.add_argument('--is_train', type=str2bool, default=True, choices=([True, False]))
        return parser

    def validate(self, opt):
        super().validate(opt)

        def require(condition, message):
            if not condition:
                self.parser.error(message)

        require(opt.is_train, 'TrainOptions requires --is_train true')
        require(opt.batch_size > 0, '--batch_size must be > 0')
        if opt.use_contrastive_loss:
            require(opt.batch_size >= 2,
                    '--use_contrastive_loss requires --batch_size >= 2')
        require(opt.num_epochs > 0, '--num_epochs must be > 0')
        require(opt.learning_rate > 0, '--learning_rate must be > 0')
        require(opt.save_freq > 0, '--save_freq must be > 0')
        require(opt.d_lr == -1.0 or opt.d_lr > 0,
                '--d_lr must be -1 (use G LR) or > 0')
        require(0.0 <= opt.ema_decay < 1.0,
                '--ema_decay must be in [0, 1)')
        require(0.0 < opt.real_label_smooth <= 1.0,
                '--real_label_smooth must be in (0, 1]')
        require(opt.d_update_every > 0, '--d_update_every must be > 0')
        require(opt.gamma >= 0, '--gamma must be >= 0')
        require(opt.lam >= 0, '--lam must be >= 0')
        require(opt.cond_warmup_epochs >= 0, '--cond_warmup_epochs must be >= 0')
        require(opt.cond_ramp_epochs >= 0, '--cond_ramp_epochs must be >= 0')

        resume_requested = opt.resume_checkpoint_path is not None or opt.resume_epoch != -1
        require(
            (opt.resume_checkpoint_path is None) == (opt.resume_epoch == -1),
            '--resume_checkpoint_path and --resume_epoch must be given together'
        )
        if resume_requested:
            require(opt.resume_epoch >= 0, '--resume_epoch must be >= 0')
            require(opt.resume_epoch < opt.num_epochs - 1,
                    '--num_epochs must leave at least one epoch after --resume_epoch')
        require(not opt.new_optim or resume_requested,
                '--new_optim is meaningful only when resuming a checkpoint')

        if opt.use_diffaugment:
            policies = [value.strip() for value in opt.diffaugment_policy.split(',')]
            valid = {'color', 'translation', 'cutout'}
            require(all(policies) and all(value in valid for value in policies),
                    '--diffaugment_policy must be a non-empty comma-separated subset of '
                    'color,translation,cutout')
