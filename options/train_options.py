from .base_options import BaseOptions

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

        # --- GAN stability levers (all opt-in; defaults reproduce the original behaviour) ---
        parser.add_argument('--d_lr', type=float, default=-1.0,
                            help='Discriminator LR (TTUR). <=0 means use --learning_rate for D too.')
        parser.add_argument('--use_ema', action='store_true',
                            help='Track an EMA of the generator weights and sample/checkpoint the EMA model.')
        parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay for the generator.')
        parser.add_argument('--real_label_smooth', type=float, default=1.0,
                            help='Discriminator real-label target (e.g. 0.9 for one-sided label smoothing).')
        parser.add_argument('--d_update_every', type=int, default=1,
                            help='Update D once every N generator steps (N>1 weakens D, i.e. n_critic<1).')
        parser.add_argument('--use_diffaugment', action='store_true',
                            help='Apply DiffAugment (same differentiable aug on BOTH real and fake) at the discriminator.')
        parser.add_argument('--diffaugment_policy', type=str, default='color,translation,cutout',
                            help='DiffAugment policy (comma-separated subset of color,translation,cutout).')

        parser.add_argument('--is_train', type=bool, default=True, choices=([True, False]))
        return parser
