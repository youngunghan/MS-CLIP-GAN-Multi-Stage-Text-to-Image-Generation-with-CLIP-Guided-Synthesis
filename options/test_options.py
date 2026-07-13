from .base_options import BaseOptions, str2bool

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # Optional: only infer.py uses --prompt; eval.py ignores it.
        parser.add_argument('--prompt', type=str, default="a photo of a person")
        parser.add_argument('--load_epoch', type=int, required=True)

        parser.add_argument('--eval_data_path', type=str, required=True)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--print_freq', type=int, default=10)
        parser.add_argument('--max_batches', type=int, default=-1,
                            help='Cap on eval batches; -1 = the whole eval set (FID needs many samples).')

        parser.add_argument('--is_train', type=str2bool, default=False, choices=([True, False]))
        return parser

    def validate(self, opt):
        super().validate(opt)

        def require(condition, message):
            if not condition:
                self.parser.error(message)

        require(not opt.is_train, 'TestOptions requires --is_train false')
        require(opt.load_epoch >= 0, '--load_epoch must be >= 0')
        require(opt.batch_size > 0, '--batch_size must be > 0')
        require(opt.print_freq > 0, '--print_freq must be > 0')
        require(opt.max_batches == -1 or opt.max_batches > 0,
                '--max_batches must be -1 or > 0')
