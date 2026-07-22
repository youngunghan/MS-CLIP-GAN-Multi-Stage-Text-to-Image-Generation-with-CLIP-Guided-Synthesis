import contextlib
import io
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from options import base_options
from options.test_options import TestOptions
from options.train_options import TrainOptions
from utils.utils import seed_fix


class TrainOptionValidationTests(unittest.TestCase):
    def _parse(self, arguments):
        with mock.patch.object(sys, 'argv', ['train.py'] + arguments):
            return TrainOptions().parse(print_options=False)

    def _assert_parse_error(self, arguments, text):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            self._parse(arguments)
        self.assertIn(text, stderr.getvalue())

    def test_fresh_correctness_modes_and_mismatch_negative_are_defaults(self):
        options = self._parse(['--gpu_ids', '-1', '--batch_size', '2'])
        self.assertEqual(options.conditioning_activation, 'linear')
        self.assertEqual(options.alignment_mode, 'image_only')
        self.assertTrue(options.use_mismatched_condition)

        options = self._parse([
            '--gpu_ids', '-1', '--batch_size', '2', '--no_mismatched_condition'
        ])
        self.assertFalse(options.use_mismatched_condition)

    def test_contrastive_singleton_and_zero_ranges_are_rejected(self):
        self._assert_parse_error(
            ['--gpu_ids', '-1', '--use_contrastive_loss', '--batch_size', '1'],
            'requires --batch_size >= 2',
        )
        self._assert_parse_error(
            ['--gpu_ids', '-1', '--save_freq', '0'], '--save_freq must be > 0'
        )
        self._assert_parse_error(
            ['--gpu_ids', '-1', '--d_update_every', '0'],
            '--d_update_every must be > 0',
        )

    def test_fixed_rgb_and_scalar_discriminator_contracts_are_rejected(self):
        self._assert_parse_error(
            ['--gpu_ids', '-1', '--g_out_chans', '4'],
            '--g_out_chans must be 3',
        )
        self._assert_parse_error(
            ['--gpu_ids', '-1', '--d_out_chans', '2'],
            '--d_out_chans must be 1',
        )

    def test_conditioning_pressure_flags_default_to_original_hardcoded_values(self):
        options = self._parse(['--gpu_ids', '-1', '--batch_size', '2'])
        self.assertEqual(options.gamma, 5.0)
        self.assertEqual(options.lam, 10.0)
        self.assertEqual(options.cond_warmup_epochs, 0)
        self.assertEqual(options.cond_ramp_epochs, 0)

    def test_conditioning_pressure_flags_are_tunable(self):
        options = self._parse([
            '--gpu_ids', '-1', '--batch_size', '2',
            '--gamma', '2.5', '--lam', '7', '--cond_warmup_epochs', '5',
            '--cond_ramp_epochs', '3',
        ])
        self.assertEqual(options.gamma, 2.5)
        self.assertEqual(options.lam, 7.0)
        self.assertEqual(options.cond_warmup_epochs, 5)
        self.assertEqual(options.cond_ramp_epochs, 3)

    def test_negative_conditioning_pressure_values_are_rejected(self):
        self._assert_parse_error(
            ['--gpu_ids', '-1', '--gamma', '-1'], '--gamma must be >= 0'
        )
        self._assert_parse_error(
            ['--gpu_ids', '-1', '--lam', '-1'], '--lam must be >= 0'
        )
        self._assert_parse_error(
            ['--gpu_ids', '-1', '--cond_warmup_epochs', '-1'],
            '--cond_warmup_epochs must be >= 0',
        )
        self._assert_parse_error(
            ['--gpu_ids', '-1', '--cond_ramp_epochs', '-1'],
            '--cond_ramp_epochs must be >= 0',
        )

    def test_seed_fix_reproducibly_seeds_python_caption_rng(self):
        seed_fix(73)
        first = [random.randint(0, 100000) for _ in range(5)]
        random.seed(999)
        seed_fix(73)
        second = [random.randint(0, 100000) for _ in range(5)]
        self.assertEqual(first, second)


class OptionPathIsolationTests(unittest.TestCase):
    def test_custom_training_checkpoint_root_is_run_namespaced(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory) / 'custom-checkpoints'
            result_path = Path(directory) / 'direct-results'
            argv = [
                'train.py', '--gpu_ids', '-1', '--batch_size', '2',
                '--name', 'audit', '--checkpoint_path', str(checkpoint_root),
                '--result_path', str(result_path),
            ]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(
                base_options, 'training_run_suffix', return_value='-STAMP'
            ), contextlib.redirect_stdout(io.StringIO()):
                options = TrainOptions().parse(print_options=True)

            run_dir = checkpoint_root / 'audit-STAMP'
            self.assertEqual(options.checkpoint_path, run_dir / 'ckpt')
            self.assertEqual(options.result_path, result_path)
            self.assertTrue((run_dir / 'ckpt').is_dir())
            self.assertTrue((run_dir / 'opt.txt').is_file())
            # Explicit result paths remain final paths and are created later by train.py.
            self.assertFalse(result_path.exists())

    def test_default_result_follows_custom_training_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory) / 'custom-checkpoints'
            argv = [
                'train.py', '--gpu_ids', '-1', '--batch_size', '2',
                '--name', 'audit', '--checkpoint_path', str(checkpoint_root),
            ]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(
                base_options, 'training_run_suffix', return_value='-STAMP'
            ), contextlib.redirect_stdout(io.StringIO()):
                options = TrainOptions().parse(print_options=True)

            run_dir = checkpoint_root / 'audit-STAMP'
            self.assertEqual(options.checkpoint_path, run_dir / 'ckpt')
            self.assertEqual(options.result_path, run_dir / 'res')
            self.assertTrue((run_dir / 'res').is_dir())

    def test_test_options_never_write_or_transform_checkpoint_input(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / 'input-checkpoint'
            checkpoint_dir.mkdir()
            sentinel = checkpoint_dir / 'epoch_3_Gen.pt'
            sentinel.write_bytes(b'checkpoint')
            before = sorted(path.relative_to(checkpoint_dir) for path in checkpoint_dir.rglob('*'))
            argv = [
                'eval.py', '--gpu_ids', '-1', '--checkpoint_path',
                str(checkpoint_dir), '--load_epoch', '3',
                '--eval_data_path', str(Path(directory) / 'eval.zip'),
            ]
            with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(
                io.StringIO()
            ):
                options = TestOptions().parse(print_options=True)

            after = sorted(path.relative_to(checkpoint_dir) for path in checkpoint_dir.rglob('*'))
            self.assertEqual(options.checkpoint_path, checkpoint_dir)
            self.assertEqual(options.name, 'experiment_name')
            self.assertEqual(before, after)

    def test_run_suffix_is_process_unique_and_does_not_consume_python_rng(self):
        random.seed(991)
        state_before = random.getstate()
        with mock.patch.object(
            base_options.time, 'time_ns', side_effect=[1234567890123456789] * 2
        ), mock.patch.object(
            base_options.os, 'getpid', side_effect=[101, 202]
        ):
            first = base_options.training_run_suffix()
            second = base_options.training_run_suffix()
        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith('-p101'))
        self.assertTrue(second.endswith('-p202'))
        self.assertEqual(random.getstate(), state_before)

    def test_existing_training_namespace_fails_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory) / 'checkpoints'
            argv = [
                'train.py', '--gpu_ids', '-1', '--batch_size', '2',
                '--name', 'same', '--checkpoint_path', str(checkpoint_root),
            ]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(
                base_options, 'training_run_suffix', return_value='-FIXED'
            ), contextlib.redirect_stdout(io.StringIO()):
                TrainOptions().parse(print_options=True)

            snapshot = checkpoint_root / 'same-FIXED' / 'opt.txt'
            original = snapshot.read_bytes()
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(
                base_options, 'training_run_suffix', return_value='-FIXED'
            ), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
                FileExistsError, 'namespace already exists'
            ):
                TrainOptions().parse(print_options=True)
            self.assertEqual(snapshot.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
