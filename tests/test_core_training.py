import types
import unittest
from unittest import mock

import torch
from torch.nn.utils import spectral_norm

from dataset.dataloader import get_dataloader
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


if __name__ == '__main__':
    unittest.main()
