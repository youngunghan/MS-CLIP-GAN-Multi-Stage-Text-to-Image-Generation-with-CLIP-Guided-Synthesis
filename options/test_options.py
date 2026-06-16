from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # Optional: only infer.py uses --prompt; eval.py ignores it.
        parser.add_argument('--prompt', type=str, default="a photo of a person")
        parser.add_argument('--load_epoch', type=int, required=True)

        parser.add_argument('--eval_data_path', type=str, required=True)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--print_freq', type=int, default=10)

        parser.add_argument('--is_train', type=bool, default=False, choices=([True, False]))
        return parser
