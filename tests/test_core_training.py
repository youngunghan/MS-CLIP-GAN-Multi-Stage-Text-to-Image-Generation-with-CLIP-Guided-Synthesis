import types
import unittest
from unittest import mock

import torch
from torch.nn.utils import spectral_norm

from dataset.dataloader import get_dataloader
from networks.discriminator import Discriminator
from networks.generator import Generator
from scripts import trainer
from scripts.trainer import frozen_discriminators, preserved_module_buffers


class _ToyDataset(torch.utils.data.Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.tensor(index)


class DataloaderContractTests(unittest.TestCase):
    def _batch_sizes(self, length, batch_size, use_contrastive_loss):
        args = types.SimpleNamespace(
            batch_size=batch_size,
            num_workers=0,
            use_contrastive_loss=use_contrastive_loss,
        )
        loader = get_dataloader(args, _ToyDataset(length), is_train=True)
        return [len(batch) for batch in loader]

    def test_only_singleton_contrastive_remainder_is_dropped(self):
        self.assertEqual(self._batch_sizes(5, 2, True), [2, 2])
        self.assertEqual(self._batch_sizes(6, 4, True), [4, 2])

    def test_noncontrastive_singleton_is_retained(self):
        self.assertEqual(self._batch_sizes(5, 2, False), [2, 2, 1])

    def test_contrastive_batch_size_one_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'batch_size >= 2'):
            self._batch_sizes(4, 1, True)


class FreezeDiscriminatorTests(unittest.TestCase):
    def test_generator_phase_changes_no_discriminator_state_or_gradients(self):
        discriminator = torch.nn.Sequential(
            spectral_norm(torch.nn.Conv2d(3, 4, 3, padding=1, bias=False)),
            torch.nn.BatchNorm2d(4),
            torch.nn.LeakyReLU(),
        )
        discriminator.train()
        # Verify non-uniform requires_grad state is restored, too.
        first_parameter = next(discriminator.parameters())
        first_parameter.requires_grad_(False)
        original_requires_grad = [p.requires_grad for p in discriminator.parameters()]
        state_before = {
            key: value.detach().clone() for key, value in discriminator.state_dict().items()
        }

        generator_output = torch.randn(2, 3, 8, 8, requires_grad=True)
        with frozen_discriminators([discriminator]):
            self.assertFalse(discriminator.training)
            self.assertTrue(all(not p.requires_grad for p in discriminator.parameters()))
            discriminator(generator_output).sum().backward()

        self.assertTrue(discriminator.training)
        self.assertEqual(
            [p.requires_grad for p in discriminator.parameters()],
            original_requires_grad,
        )
        self.assertTrue(all(p.grad is None for p in discriminator.parameters()))
        self.assertIsNotNone(generator_output.grad)
        for key, value in discriminator.state_dict().items():
            torch.testing.assert_close(value, state_before[key], msg=lambda msg: f'{key}: {msg}')


class GeneratorBufferIsolationTests(unittest.TestCase):
    def test_d_only_forward_is_state_neutral_but_keeps_train_batch_stats(self):
        generator = torch.nn.Sequential(
            torch.nn.Linear(3, 3, bias=False),
            torch.nn.BatchNorm1d(3),
        )
        with torch.no_grad():
            generator[0].weight.copy_(torch.eye(3))
        generator.train()
        input_tensor = torch.tensor([
            [1.0, 3.0, 5.0],
            [2.0, 5.0, 8.0],
            [4.0, 7.0, 11.0],
            [8.0, 9.0, 14.0],
        ])
        state_before = {
            key: value.detach().clone() for key, value in generator.state_dict().items()
        }
        modes_before = [module.training for module in generator.modules()]

        with torch.no_grad(), preserved_module_buffers(generator):
            d_only_output = generator(input_tensor)

        # Train-mode BatchNorm uses this batch, so each output feature is centered.
        torch.testing.assert_close(
            d_only_output.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0.0
        )
        self.assertEqual(
            [module.training for module in generator.modules()], modes_before
        )
        for key, value in generator.state_dict().items():
            torch.testing.assert_close(value, state_before[key], msg=lambda msg: f'{key}: {msg}')

        # The subsequent normal G-phase forward advances BN state exactly once.
        generator(input_tensor)
        batch_norm = generator[1]
        self.assertEqual(
            batch_norm.num_batches_tracked.item(),
            state_before['1.num_batches_tracked'].item() + 1,
        )
        self.assertFalse(torch.equal(batch_norm.running_mean, state_before['1.running_mean']))


class LazyLossWarmupTests(unittest.TestCase):
    def test_vgg_is_materialized_only_when_mixed_loss_is_enabled(self):
        with mock.patch.object(trainer, 'get_vgg_perceptual_loss') as get_vgg:
            trainer.warmup_training_losses(False, torch.device('cpu'))
            get_vgg.assert_not_called()
            trainer.warmup_training_losses(True, torch.device('cpu'))
            get_vgg.assert_called_once_with(torch.device('cpu'))


# Shared tiny single-stage Generator/Discriminator dimensions for the real (non-mocked)
# forward/training-step coverage below. curr_stage=0 / img size 64 mirrors the
# smallest real config in tests/test_core_models.py's DetailedDiscriminatorForwardTests.
_TINY_CLIP_EMB_DIM = 4
_TINY_COND_DIM = 2
_TINY_NOISE_DIM = 4
_TINY_IMG_SIZE = 64


def _build_tiny_generator(device):
    return Generator(
        in_chans=16, out_chans=3, noise_dim=_TINY_NOISE_DIM, cond_dim=_TINY_COND_DIM,
        clip_emb_dim=_TINY_CLIP_EMB_DIM, num_stage=1, device=device,
    )


def _build_tiny_discriminator(device):
    return Discriminator(
        img_chans=3, in_chans=1, out_chans=1, condition_dim=_TINY_COND_DIM,
        clip_text_embedding_dim=_TINY_CLIP_EMB_DIM, curr_stage=0, device=device,
        alignment_mode='image_only',
    )


class _TinyRealDataset(torch.utils.data.Dataset):
    """Minimal real (non-mocked) multi-stage sample source for a train_step smoke test.

    Mirrors dataset.dataloader.MM_CelebA's __getitem__ contract: a list of
    per-stage images (ascending resolution), a CLIP image embedding, and a CLIP
    text embedding.
    """

    def __init__(self, length, clip_emb_dim, img_size):
        self.length = length
        self.clip_emb_dim = clip_emb_dim
        self.img_size = img_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = torch.randn(3, self.img_size, self.img_size)
        img_feature = torch.randn(self.clip_emb_dim)
        txt_feature = torch.randn(self.clip_emb_dim)
        return [image], img_feature, txt_feature


class _StubWriter:
    """A do-nothing stand-in for SummaryWriter.add_scalar; TensorBoard I/O is not under test."""

    def add_scalar(self, *args, **kwargs):
        pass


class RealTrainStepSmokeTest(unittest.TestCase):
    def test_single_stage_train_step_runs_and_updates_params_with_finite_losses(self):
        torch.manual_seed(0)
        device = torch.device('cpu')
        batch_size = 4

        model_G = _build_tiny_generator(device)
        model_D = _build_tiny_discriminator(device)

        dataset = _TinyRealDataset(
            length=batch_size, clip_emb_dim=_TINY_CLIP_EMB_DIM, img_size=_TINY_IMG_SIZE
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        optim_g = torch.optim.Adam(model_G.parameters(), lr=1e-3)
        optim_d = torch.optim.Adam(model_D.parameters(), lr=1e-3)

        g_params_before = [p.detach().clone() for p in model_G.parameters()]
        d_params_before = [p.detach().clone() for p in model_D.parameters()]

        d_loss_epoch, g_loss_epoch, save_txt_feature = trainer.train_step(
            train_loader=train_loader, noise_dim=_TINY_NOISE_DIM, model_G=model_G,
            model_D_lst=[model_D], optim_g=optim_g, optim_d_lst=[optim_d],
            loss_fn=torch.nn.BCELoss(), num_stage=1,
            use_uncond_loss=False, use_contrastive_loss=False, use_mixed_loss=False,
            clip_model=None, gamma=1.0, lam=1.0, report_interval=10,
            device=device, epoch=0, writer=_StubWriter(),
        )

        self.assertTrue(torch.isfinite(torch.tensor(d_loss_epoch)))
        self.assertTrue(torch.isfinite(torch.tensor(g_loss_epoch)))
        self.assertEqual(tuple(save_txt_feature.shape), (batch_size, _TINY_CLIP_EMB_DIM))

        # Both the generator and discriminator must have taken a real optimizer
        # step (not merely produced a loss number) for this to be a meaningful
        # training-step smoke test.
        self.assertTrue(any(
            not torch.equal(before, after.detach())
            for before, after in zip(g_params_before, model_G.parameters())
        ))
        self.assertTrue(any(
            not torch.equal(before, after.detach())
            for before, after in zip(d_params_before, model_D.parameters())
        ))


class ConditioningWarmupWiringTests(unittest.TestCase):
    """train_step must thread --cond_warmup_epochs/--cond_ramp_epochs into D_loss's
    cond_gate via criteria.loss.conditioning_gate(epoch, ...), not silently drop them.
    """

    def _cond_gate_seen_by_d_loss(self, epoch, cond_warmup_epochs, cond_ramp_epochs):
        torch.manual_seed(0)
        device = torch.device('cpu')
        batch_size = 4

        model_G = _build_tiny_generator(device)
        model_D = _build_tiny_discriminator(device)
        dataset = _TinyRealDataset(
            length=batch_size, clip_emb_dim=_TINY_CLIP_EMB_DIM, img_size=_TINY_IMG_SIZE
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        optim_g = torch.optim.Adam(model_G.parameters(), lr=1e-3)
        optim_d = torch.optim.Adam(model_D.parameters(), lr=1e-3)

        with mock.patch.object(trainer, 'D_loss', wraps=trainer.D_loss) as spy:
            trainer.train_step(
                train_loader=train_loader, noise_dim=_TINY_NOISE_DIM, model_G=model_G,
                model_D_lst=[model_D], optim_g=optim_g, optim_d_lst=[optim_d],
                loss_fn=torch.nn.BCELoss(), num_stage=1,
                use_uncond_loss=False, use_contrastive_loss=False, use_mixed_loss=False,
                clip_model=None, gamma=1.0, lam=1.0, report_interval=10,
                device=device, epoch=epoch, writer=_StubWriter(),
                cond_warmup_epochs=cond_warmup_epochs, cond_ramp_epochs=cond_ramp_epochs,
            )
        self.assertTrue(spy.called)
        return spy.call_args.kwargs['cond_gate']

    def test_cond_gate_is_zero_during_warmup_epoch(self):
        cond_gate = self._cond_gate_seen_by_d_loss(
            epoch=0, cond_warmup_epochs=3, cond_ramp_epochs=0
        )
        self.assertEqual(cond_gate, 0.0)

    def test_cond_gate_is_one_once_warmup_ends(self):
        cond_gate = self._cond_gate_seen_by_d_loss(
            epoch=3, cond_warmup_epochs=3, cond_ramp_epochs=0
        )
        self.assertEqual(cond_gate, 1.0)

    def test_default_cond_gate_is_always_one(self):
        cond_gate = self._cond_gate_seen_by_d_loss(
            epoch=0, cond_warmup_epochs=0, cond_ramp_epochs=0
        )
        self.assertEqual(cond_gate, 1.0)


class RealGeneratorForwardTest(unittest.TestCase):
    def test_forward_produces_finite_correctly_shaped_stage_output(self):
        torch.manual_seed(0)
        device = torch.device('cpu')
        batch_size = 3

        model_G = _build_tiny_generator(device)
        txt_feature = torch.randn(batch_size, _TINY_CLIP_EMB_DIM)
        noise = torch.randn(batch_size, _TINY_NOISE_DIM)

        fake_images, mu, log_sigma = model_G(txt_feature, noise)

        self.assertEqual(len(fake_images), 1)
        self.assertEqual(
            tuple(fake_images[0].shape), (batch_size, 3, _TINY_IMG_SIZE, _TINY_IMG_SIZE)
        )
        self.assertEqual(tuple(mu.shape), (batch_size, _TINY_COND_DIM))
        self.assertEqual(tuple(log_sigma.shape), (batch_size, _TINY_COND_DIM))
        self.assertTrue(torch.isfinite(fake_images[0]).all())
        self.assertTrue(torch.isfinite(mu).all())
        self.assertTrue(torch.isfinite(log_sigma).all())


if __name__ == '__main__':
    unittest.main()
