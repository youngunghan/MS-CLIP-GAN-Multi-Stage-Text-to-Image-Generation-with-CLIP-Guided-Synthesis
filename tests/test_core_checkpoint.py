import os
import random
import hashlib
import tempfile
import types
import unittest

import numpy as np
import torch

from utils.utils import (
    CHECKPOINT_FORMAT_VERSION,
    build_training_provenance,
    load_checkpoint,
    peek_checkpoint_metadata,
    save_checkpoint,
    scheduler_horizon,
)


class _TinyGenerator(torch.nn.Module):
    def __init__(self, conditioning_activation='linear'):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
        self.in_chans = 32
        self.out_chans = 3
        self.noise_dim = 4
        self.cond_dim = 2
        self.c_txt_dim = 8
        self.num_stage = 1
        self.conditioning_activation = conditioning_activation

    def set_conditioning_activation(self, activation):
        self.conditioning_activation = activation


class _TinyDiscriminator(torch.nn.Module):
    def __init__(self, alignment_mode='image_only'):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
        self.alignment_mode = alignment_mode

    def set_alignment_mode(self, mode):
        self.alignment_mode = mode


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.args = types.SimpleNamespace(
            checkpoint_path=self.temp_dir.name,
            alignment_mode='image_only',
            is_train=True,
            new_optim=False,
            num_epochs=5,
            seed=17,
            use_uncond_loss=False,
            use_contrastive_loss=False,
            use_mixed_loss=False,
            use_mismatched_condition=True,
            use_diffaugment=False,
            diffaugment_policy='color,translation,cutout',
            d_update_every=1,
            real_label_smooth=1.0,
            use_ema=False,
            ema_decay=0.999,
            batch_size=2,
            num_workers=0,
            save_freq=1,
            training_provenance={
                'schema_version': 1,
                'dataset': {'sha256': 'data-hash', 'size_bytes': 123},
                'source': {'sha256': 'source-hash', 'files': {}},
                'git': {'commit': 'deadbeef', 'dirty': False},
                'runtime': {'python': 'test', 'torch': 'test'},
                'hardware': {'machine': 'test'},
            },
        )

    @staticmethod
    def _training_objects(conditioning='linear', alignment='image_only', horizon=5):
        generator = _TinyGenerator(conditioning)
        discriminator = _TinyDiscriminator(alignment)
        optim_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
        optim_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=horizon)
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=horizon)
        return generator, discriminator, optim_g, optim_d, sched_g, sched_d

    @staticmethod
    def _advance_schedulers(optim_g, optim_d, sched_g, sched_d, steps):
        for _ in range(steps):
            optim_g.step()
            optim_d.step()
            sched_g.step()
            sched_d.step()

    def _save(self, g_raw=None):
        g, d, optim_g, optim_d, sched_g, sched_d = self._training_objects()
        self._advance_schedulers(optim_g, optim_d, sched_g, sched_d, steps=2)
        save_checkpoint(
            self.args, g, [d], optim_g, [optim_d], epoch=1, num_stage=1,
            scheduler_g=sched_g, scheduler_d_lst=[sched_d], g_raw=g_raw,
        )
        return g, d

    def test_v2_metadata_and_rng_round_trip(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        saved_g, saved_d = self._save()
        metadata = peek_checkpoint_metadata(self.temp_dir.name, 1)

        self.assertEqual(metadata['format_version'], CHECKPOINT_FORMAT_VERSION)
        self.assertFalse(metadata['legacy'])
        self.assertEqual(metadata['model_config']['conditioning_activation'], 'linear')
        self.assertEqual(metadata['model_config']['alignment_mode'], 'image_only')
        self.assertEqual(metadata['training_config']['batch_size'], 2)
        self.assertEqual(metadata['training_config']['seed'], 17)
        self.assertTrue(metadata['training_config']['use_mismatched_condition'])
        self.assertEqual(metadata['training_config']['target_num_epochs'], 5)
        # The conditioning-pressure schedule (--gamma/--lam/--cond_warmup_epochs/
        # --cond_ramp_epochs) is recorded in training_config just like the other
        # training options, so a run's schedule is recoverable from its checkpoint.
        # self.args does not set these, so they fall back to the original
        # hardcoded values (getattr default) -- proving the fallback also flows
        # through into the saved provenance.
        self.assertEqual(metadata['training_config']['gamma'], 5.0)
        self.assertEqual(metadata['training_config']['lam'], 10.0)
        self.assertEqual(metadata['training_config']['cond_warmup_epochs'], 0)
        self.assertEqual(metadata['training_config']['cond_ramp_epochs'], 0)
        self.assertEqual(
            metadata['schedule_config'],
            {'phase_start_epoch': 0, 'phase_end_epoch': 5, 't_max': 5,
             'scheduler_type': 'CosineAnnealingLR'},
        )
        self.assertTrue(metadata['rng_state_available'])
        self.assertEqual(
            metadata['training_provenance']['dataset']['sha256'], 'data-hash'
        )

        expected_python = random.random()
        expected_numpy = np.random.rand()
        expected_torch = torch.rand(3)
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)

        g, d, optim_g, optim_d, sched_g, sched_d = self._training_objects(
            conditioning='relu', alignment='legacy_conditioned'
        )
        load_checkpoint(
            self.args, g, [d], optim_g, [optim_d], self.temp_dir.name, 1,
            scheduler_g=sched_g, scheduler_d_lst=[sched_d],
        )

        self.assertEqual(g.conditioning_activation, 'linear')
        self.assertEqual(d.alignment_mode, 'image_only')
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.rand(), expected_numpy)
        torch.testing.assert_close(torch.rand(3), expected_torch)
        for actual, expected in zip(g.parameters(), saved_g.parameters()):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(d.parameters(), saved_d.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_legacy_checkpoint_automatically_selects_legacy_modes(self):
        g, d, optim_g, optim_d, _, _ = self._training_objects()
        torch.save(
            {'model': g.state_dict(), 'optimizer': optim_g.state_dict(),
             'scheduler': None, 'epoch': 2, 'num_stage': 1},
            os.path.join(self.temp_dir.name, 'epoch_2_Gen.pt'),
        )
        torch.save(
            {'model': d.state_dict(), 'optimizer': optim_d.state_dict(),
             'scheduler': None, 'epoch': 2, 'num_stage': 1},
            os.path.join(self.temp_dir.name, 'epoch_2_Dis_0.pt'),
        )
        inference_args = types.SimpleNamespace(is_train=False, new_optim=False)
        load_checkpoint(
            inference_args, g, [d], None, [None], self.temp_dir.name, 2
        )
        self.assertEqual(g.conditioning_activation, 'relu')
        self.assertEqual(d.alignment_mode, 'legacy_conditioned')

    def test_legacy_optimizer_resume_is_best_effort_with_explicit_warning(self):
        g, d, optim_g, optim_d, sched_g, sched_d = self._training_objects()
        self._advance_schedulers(optim_g, optim_d, sched_g, sched_d, steps=2)
        torch.save(
            {'model': g.state_dict(), 'optimizer': optim_g.state_dict(),
             'scheduler': sched_g.state_dict(), 'epoch': 2, 'num_stage': 1},
            os.path.join(self.temp_dir.name, 'epoch_2_Gen.pt'),
        )
        torch.save(
            {'model': d.state_dict(), 'optimizer': optim_d.state_dict(),
             'scheduler': sched_d.state_dict(), 'epoch': 2, 'num_stage': 1},
            os.path.join(self.temp_dir.name, 'epoch_2_Dis_0.pt'),
        )
        target_g, target_d, target_og, target_od, target_sg, target_sd = (
            self._training_objects()
        )
        with self.assertWarnsRegex(RuntimeWarning, 'best-effort legacy resume'):
            load_checkpoint(
                self.args, target_g, [target_d], target_og, [target_od],
                self.temp_dir.name, 2, scheduler_g=target_sg,
                scheduler_d_lst=[target_sd],
            )
        self.assertEqual(target_g.conditioning_activation, 'relu')
        self.assertEqual(target_d.alignment_mode, 'legacy_conditioned')

    def test_legacy_training_without_scheduler_requires_new_phase(self):
        g, d, optim_g, optim_d, _, _ = self._training_objects()
        torch.save(
            {'model': g.state_dict(), 'optimizer': optim_g.state_dict(),
             'scheduler': None, 'epoch': 2, 'num_stage': 1},
            os.path.join(self.temp_dir.name, 'epoch_2_Gen.pt'),
        )
        torch.save(
            {'model': d.state_dict(), 'optimizer': optim_d.state_dict(),
             'scheduler': None, 'epoch': 2, 'num_stage': 1},
            os.path.join(self.temp_dir.name, 'epoch_2_Dis_0.pt'),
        )
        target_g, target_d, target_og, target_od, target_sg, target_sd = (
            self._training_objects()
        )
        with self.assertRaisesRegex(ValueError, 'use --new_optim'):
            load_checkpoint(
                self.args, target_g, [target_d], target_og, [target_od],
                self.temp_dir.name, 2, scheduler_g=target_sg,
                scheduler_d_lst=[target_sd],
            )

    def test_exact_resume_rejects_changed_horizon_and_missing_scheduler_object(self):
        self._save()
        g, d, optim_g, optim_d, sched_g, sched_d = self._training_objects(horizon=7)
        changed_horizon_args = types.SimpleNamespace(**vars(self.args))
        changed_horizon_args.num_epochs = 7
        with self.assertRaisesRegex(ValueError, 'phase_end_epoch=5'):
            load_checkpoint(
                changed_horizon_args, g, [d], optim_g, [optim_d],
                self.temp_dir.name, 1,
                scheduler_g=sched_g, scheduler_d_lst=[sched_d],
            )

        g, d, optim_g, optim_d, _, sched_d = self._training_objects()
        with self.assertRaisesRegex(ValueError, 'current generator scheduler'):
            load_checkpoint(
                self.args, g, [d], optim_g, [optim_d], self.temp_dir.name, 1,
                scheduler_g=None, scheduler_d_lst=[sched_d],
            )

    def test_ema_marker_requires_raw_companion_for_exact_but_not_new_phase(self):
        ema, _, _, _, _, _ = self._training_objects()
        raw, d, optim_g, optim_d, sched_g, sched_d = self._training_objects()
        ema_args = types.SimpleNamespace(**vars(self.args))
        ema_args.use_ema = True
        self._advance_schedulers(optim_g, optim_d, sched_g, sched_d, steps=4)
        save_checkpoint(
            ema_args, ema, [d], optim_g, [optim_d], epoch=3, num_stage=1,
            scheduler_g=sched_g, scheduler_d_lst=[sched_d], g_raw=raw,
        )
        os.remove(os.path.join(self.temp_dir.name, 'epoch_3_Gen_raw.pt'))

        target_g, target_d, target_og, target_od, target_sg, target_sd = (
            self._training_objects()
        )
        target_ema = _TinyGenerator()
        with self.assertRaisesRegex(FileNotFoundError, 'required raw companion'):
            load_checkpoint(
                ema_args, target_g, [target_d], target_og, [target_od],
                self.temp_dir.name, 3,
                scheduler_g=target_sg, scheduler_d_lst=[target_sd],
                g_ema=target_ema,
            )

        # --new_optim explicitly starts from the available primary weights and may
        # change EMA policy; it does not claim exact continuation.
        new_phase_args = types.SimpleNamespace(**vars(self.args))
        new_phase_args.new_optim = True
        load_checkpoint(
            new_phase_args, target_g, [target_d], target_og, [target_od],
            self.temp_dir.name, 3,
        )

    def test_exact_resume_compares_training_config_and_ema_semantics(self):
        self._save()
        g, d, optim_g, optim_d, sched_g, sched_d = self._training_objects()
        changed = types.SimpleNamespace(**vars(self.args))
        changed.save_freq = 2
        with self.assertRaisesRegex(ValueError, 'save_freq'):
            load_checkpoint(
                changed, g, [d], optim_g, [optim_d], self.temp_dir.name, 1,
                scheduler_g=sched_g, scheduler_d_lst=[sched_d],
            )

        changed_warmup = types.SimpleNamespace(**vars(self.args))
        changed_warmup.cond_warmup_epochs = 5
        with self.assertRaisesRegex(ValueError, 'cond_warmup_epochs'):
            load_checkpoint(
                changed_warmup, g, [d], optim_g, [optim_d], self.temp_dir.name, 1,
                scheduler_g=sched_g, scheduler_d_lst=[sched_d],
            )

        changed_data = types.SimpleNamespace(**vars(self.args))
        changed_data.training_provenance = {
            **self.args.training_provenance,
            'dataset': {'sha256': 'different', 'size_bytes': 123},
        }
        with self.assertRaisesRegex(ValueError, 'dataset.sha256'):
            load_checkpoint(
                changed_data, g, [d], optim_g, [optim_d], self.temp_dir.name, 1,
                scheduler_g=sched_g, scheduler_d_lst=[sched_d],
            )

        enable_ema = types.SimpleNamespace(**vars(self.args))
        enable_ema.use_ema = True
        with self.assertRaisesRegex(ValueError, 'use_ema'):
            load_checkpoint(
                enable_ema, g, [d], optim_g, [optim_d], self.temp_dir.name, 1,
                scheduler_g=sched_g, scheduler_d_lst=[sched_d],
                g_ema=_TinyGenerator(),
            )

        ema, _, _, _, _, _ = self._training_objects()
        raw, d, optim_g, optim_d, sched_g, sched_d = self._training_objects()
        ema_args = types.SimpleNamespace(**vars(self.args))
        ema_args.use_ema = True
        self._advance_schedulers(optim_g, optim_d, sched_g, sched_d, steps=4)
        save_checkpoint(
            ema_args, ema, [d], optim_g, [optim_d], epoch=3, num_stage=1,
            scheduler_g=sched_g, scheduler_d_lst=[sched_d], g_raw=raw,
        )
        current_g, current_d, current_og, current_od, current_sg, current_sd = (
            self._training_objects()
        )
        current_ema = _TinyGenerator()
        changed_decay = types.SimpleNamespace(**vars(ema_args))
        changed_decay.ema_decay = 0.9
        with self.assertRaisesRegex(ValueError, 'ema_decay'):
            load_checkpoint(
                changed_decay, current_g, [current_d], current_og, [current_od],
                self.temp_dir.name, 3, scheduler_g=current_sg,
                scheduler_d_lst=[current_sd], g_ema=current_ema,
            )

        disable_ema = types.SimpleNamespace(**vars(self.args))
        with self.assertRaisesRegex(ValueError, 'use_ema'):
            load_checkpoint(
                disable_ema, current_g, [current_d], current_og, [current_od],
                self.temp_dir.name, 3, scheduler_g=current_sg,
                scheduler_d_lst=[current_sd], g_ema=None,
            )

    def test_interrupted_extended_phase_reuses_saved_t_max(self):
        extension_args = types.SimpleNamespace(**vars(self.args))
        extension_args.num_epochs = 200
        g, d, optim_g, optim_d, sched_g, sched_d = self._training_objects(horizon=50)
        self._advance_schedulers(optim_g, optim_d, sched_g, sched_d, steps=11)
        save_checkpoint(
            extension_args, g, [d], optim_g, [optim_d], epoch=160, num_stage=1,
            scheduler_g=sched_g, scheduler_d_lst=[sched_d],
        )
        metadata = peek_checkpoint_metadata(self.temp_dir.name, 160)
        self.assertEqual(
            metadata['schedule_config'],
            {'phase_start_epoch': 150, 'phase_end_epoch': 200, 't_max': 50,
             'scheduler_type': 'CosineAnnealingLR'},
        )
        restored_horizon = scheduler_horizon(
            200, resume_epoch=160, new_optim=False,
            schedule_config=metadata['schedule_config'],
        )
        self.assertEqual(restored_horizon, 50)

        target_g, target_d, target_og, target_od, target_sg, target_sd = (
            self._training_objects(horizon=restored_horizon)
        )
        load_checkpoint(
            extension_args, target_g, [target_d], target_og, [target_od],
            self.temp_dir.name, 160, scheduler_g=target_sg,
            scheduler_d_lst=[target_sd],
        )

        with self.assertRaisesRegex(ValueError, 'phase ends at epoch 200'):
            scheduler_horizon(
                201, resume_epoch=160, new_optim=False,
                schedule_config=metadata['schedule_config'],
            )

    def test_scheduler_progress_must_match_checkpoint_epoch(self):
        g, d, optim_g, optim_d, sched_g, sched_d = self._training_objects()
        with self.assertRaisesRegex(ValueError, 'expected 2'):
            save_checkpoint(
                self.args, g, [d], optim_g, [optim_d], epoch=1, num_stage=1,
                scheduler_g=sched_g, scheduler_d_lst=[sched_d],
            )

        self._advance_schedulers(optim_g, optim_d, sched_g, sched_d, steps=2)
        save_checkpoint(
            self.args, g, [d], optim_g, [optim_d], epoch=1, num_stage=1,
            scheduler_g=sched_g, scheduler_d_lst=[sched_d],
        )
        path = os.path.join(self.temp_dir.name, 'epoch_1_Gen.pt')
        state = torch.load(path, map_location='cpu', weights_only=True)
        state['scheduler']['last_epoch'] = 0
        torch.save(state, path)

        target_g, target_d, target_og, target_od, target_sg, target_sd = (
            self._training_objects()
        )
        with self.assertRaisesRegex(ValueError, 'inconsistent with expected 2'):
            load_checkpoint(
                self.args, target_g, [target_d], target_og, [target_od],
                self.temp_dir.name, 1, scheduler_g=target_sg,
                scheduler_d_lst=[target_sd],
            )

    def test_training_provenance_hashes_actual_data_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.join(directory, 'repo')
            os.makedirs(os.path.join(root, 'utils'))
            source_path = os.path.join(root, 'utils', 'utils.py')
            data_path = os.path.join(directory, 'train.zip')
            with open(source_path, 'wb') as handle:
                handle.write(b'print("train")\n')
            with open(data_path, 'wb') as handle:
                handle.write(b'dataset-bytes')

            provenance = build_training_provenance(
                data_path, device_ids=[], project_root=root
            )
            self.assertEqual(
                provenance['dataset']['sha256'],
                hashlib.sha256(b'dataset-bytes').hexdigest(),
            )
            self.assertEqual(
                provenance['source']['files']['utils/utils.py'],
                hashlib.sha256(b'print("train")\n').hexdigest(),
            )

    def test_new_optimizer_schedule_uses_remaining_epochs(self):
        self.assertEqual(scheduler_horizon(100, resume_epoch=49, new_optim=True), 50)
        self.assertEqual(scheduler_horizon(100, resume_epoch=49, new_optim=False), 100)
        with self.assertRaisesRegex(ValueError, 'no training epochs remain'):
            scheduler_horizon(10, resume_epoch=9, new_optim=True)


if __name__ == '__main__':
    unittest.main()
