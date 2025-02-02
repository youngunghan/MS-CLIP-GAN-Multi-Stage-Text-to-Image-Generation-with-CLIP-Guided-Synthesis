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
        
        parser.add_argument('--is_train', type=bool, default=True, choices=([True, False]))
        return parser
