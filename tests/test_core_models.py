import unittest

import torch

from criteria.loss import D_loss, G_loss, contrastive_loss_D
from networks.discriminator import AlignCondDiscriminator, Discriminator
from networks.generator import ConditioningAugmention


class ConditioningAugmentationTests(unittest.TestCase):
    def test_linear_is_unconstrained_and_state_dict_is_legacy_compatible(self):
        linear = ConditioningAugmention(2, 1, torch.device('cpu'), activation='linear')
        legacy = ConditioningAugmention(2, 1, torch.device('cpu'), activation='relu')
        with torch.no_grad():
            linear.layer[0].weight.copy_(torch.tensor([[-1.0, -1.0], [-2.0, -2.0]]))
        legacy.load_state_dict(linear.state_dict(), strict=True)

        x = torch.ones(1, 2)
        _, linear_mu, linear_log_sigma = linear(x)
        _, legacy_mu, legacy_log_sigma = legacy(x)

        self.assertLess(linear_mu.item(), 0.0)
        self.assertLess(linear_log_sigma.item(), 0.0)
        self.assertEqual(legacy_mu.item(), 0.0)
        self.assertEqual(legacy_log_sigma.item(), 0.0)
        self.assertEqual(set(linear.state_dict()), set(legacy.state_dict()))


class AlignmentTests(unittest.TestCase):
    @staticmethod
    def _configured_head(mode):
        head = AlignCondDiscriminator(
            in_chans=1, cond_dim=2, text_emb_dim=1, alignment_mode=mode
        ).eval()
        with torch.no_grad():
            first_conv = head.align_net[0][0]
            first_conv.weight.zero_()
            # Depend only on the two condition channels in legacy mode.
            first_conv.weight[:, 8:, :, :].fill_(1.0)
            head.align_net[0][1].weight.fill_(1.0)
            head.align_net[0][1].bias.zero_()
            head.align_net[1][0].weight.fill_(1.0)
        return head

    def test_image_only_alignment_cannot_observe_condition(self):
        head = self._configured_head('image_only')
        image_features = torch.randn(2, 8, 4, 4)
        zeros = torch.zeros(2, 2)
        ones = torch.ones(2, 2)

        torch.testing.assert_close(head(image_features, zeros), head(image_features, ones))

    def test_legacy_alignment_retains_condition_and_shapes(self):
        image_only = self._configured_head('image_only')
        legacy = self._configured_head('legacy_conditioned')
        legacy.load_state_dict(image_only.state_dict(), strict=True)
        image_features = torch.zeros(2, 8, 4, 4)

        output_zero = legacy(image_features, torch.zeros(2, 2))
        output_one = legacy(image_features, torch.ones(2, 2))
        self.assertFalse(torch.equal(output_zero, output_one))
        self.assertEqual(set(image_only.state_dict()), set(legacy.state_dict()))


class _RecordingDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conditions = []
        self.alignment_requests = []
        self.forward_calls = 0

    def forward(self, img, condition=None, compute_alignment=True,
                mismatched_condition=None, compute_unconditional=False,
                return_details=False):
        self.forward_calls += 1
        if condition is not None:
            self.conditions.append(condition.detach().clone())
            self.alignment_requests.append(compute_alignment)
            logits = condition.sum(dim=1) * 0.0
        else:
            logits = img.flatten(1).sum(dim=1) * 0.0
        output = torch.sigmoid(logits)
        if not return_details:
            return output, None

        details = {'conditional': output}
        if mismatched_condition is not None:
            self.conditions.append(mismatched_condition.detach().clone())
            self.alignment_requests.append(False)
            details['mismatched_conditional'] = torch.sigmoid(
                mismatched_condition.sum(dim=1) * 0.0
            )
        if compute_unconditional:
            details['unconditional'] = torch.sigmoid(
                img.flatten(1).sum(dim=1) * 0.0
            )
        if compute_alignment:
            details['alignment'] = torch.zeros_like(condition)
        return details


class DiscriminatorLossTests(unittest.TestCase):
    def test_real_mismatched_text_is_default_negative_for_non_singleton(self):
        model = _RecordingDiscriminator()
        batch_size = 3
        real = torch.randn(batch_size, 3, 4, 4)
        fake = torch.randn_like(real)
        mu = torch.arange(batch_size * 2, dtype=torch.float32).view(batch_size, 2)
        labels_real = torch.ones(batch_size)
        labels_fake = torch.zeros(batch_size)

        loss = D_loss(
            real, fake, model, torch.nn.BCELoss(),
            use_uncond_loss=False, use_contrastive_loss=False,
            gamma=1.0, mu=mu, txt_feature=mu,
            d_fake_label=labels_fake, d_real_label=labels_real,
        )

        self.assertEqual(loss.ndim, 0)
        self.assertEqual(len(model.conditions), 3)
        self.assertEqual(model.forward_calls, 2)
        torch.testing.assert_close(model.conditions[-1], mu.roll(1, 0))
        self.assertEqual(model.alignment_requests, [False, False, False])

        # This recording D emits p=0.5 for every pairing. The new wrong-text
        # negative shares the old negative mass with generated fake, so merely
        # adding it does not increase the conditional loss scale by 50%.
        no_mismatch = D_loss(
            real, fake, _RecordingDiscriminator(), torch.nn.BCELoss(),
            use_uncond_loss=False, use_contrastive_loss=False,
            gamma=1.0, mu=mu, txt_feature=mu,
            d_fake_label=labels_fake, d_real_label=labels_real,
            use_mismatched_condition=False,
        )
        torch.testing.assert_close(loss, no_mismatch)

    def test_mismatched_negative_can_be_disabled(self):
        model = _RecordingDiscriminator()
        batch_size = 2
        image = torch.randn(batch_size, 3, 4, 4)
        mu = torch.randn(batch_size, 2)
        D_loss(
            image, image, model, torch.nn.BCELoss(),
            use_uncond_loss=False, use_contrastive_loss=False,
            gamma=1.0, mu=mu, txt_feature=mu,
            d_fake_label=torch.zeros(batch_size),
            d_real_label=torch.ones(batch_size),
            use_mismatched_condition=False,
        )
        self.assertEqual(len(model.conditions), 2)
        self.assertEqual(model.forward_calls, 2)

    def test_contrastive_loss_rejects_singleton(self):
        with self.assertRaisesRegex(ValueError, 'at least two'):
            contrastive_loss_D(torch.randn(1, 4), torch.randn(1, 4))


class DetailedDiscriminatorForwardTests(unittest.TestCase):
    def setUp(self):
        self.model = Discriminator(
            img_chans=3, in_chans=1, out_chans=1,
            condition_dim=2, clip_text_embedding_dim=4,
            curr_stage=0, device=torch.device('cpu'),
            alignment_mode='image_only',
        )
        self.real = torch.randn(2, 3, 64, 64)
        self.fake = torch.randn_like(self.real)
        self.mu = torch.randn(2, 2)
        self.txt = torch.randn(2, 4)
        self.loss_fn = torch.nn.BCELoss()

    def _feature_counter(self):
        calls = []
        handle = self.model.feature_net.register_forward_hook(
            lambda *_args: calls.append(1)
        )
        self.addCleanup(handle.remove)
        return calls

    def test_detailed_output_computes_all_heads_from_one_feature_forward(self):
        calls = self._feature_counter()
        details = self.model(
            self.real, condition=self.mu,
            mismatched_condition=self.mu.roll(1, 0),
            compute_unconditional=True, compute_alignment=True,
            return_details=True,
        )
        self.assertEqual(
            set(details),
            {'conditional', 'mismatched_conditional', 'unconditional', 'alignment'},
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(details['conditional'].shape, (2,))
        self.assertEqual(details['alignment'].shape, (2, 4))
        sum(value.float().sum() for value in details.values()).backward()
        self.assertTrue(any(
            parameter.grad is not None for parameter in self.model.parameters()
        ))

    def test_mismatch_toggle_does_not_add_feature_extractor_forwards(self):
        for use_mismatch in (True, False):
            calls = []
            handle = self.model.feature_net.register_forward_hook(
                lambda *_args: calls.append(1)
            )
            try:
                self.model.zero_grad(set_to_none=True)
                loss = D_loss(
                    self.real, self.fake, self.model, self.loss_fn,
                    use_uncond_loss=True, use_contrastive_loss=False,
                    gamma=1.0, mu=self.mu, txt_feature=self.txt,
                    d_fake_label=torch.zeros(2), d_real_label=torch.ones(2),
                    use_mismatched_condition=use_mismatch,
                )
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(len(calls), 2)  # one fake view + one real view
                loss.backward()
                self.assertTrue(any(
                    parameter.grad is not None for parameter in self.model.parameters()
                ))
            finally:
                handle.remove()

    def test_generator_loss_uses_one_fake_feature_forward(self):
        calls = self._feature_counter()
        fake = self.fake.detach().requires_grad_(True)
        loss = G_loss(
            self.real, fake, self.model, self.loss_fn,
            use_uncond_loss=True, use_contrastive_loss=False,
            use_mixed_loss=False, clip_model=None, gamma=1.0, lam=1.0,
            mu=self.mu, txt_feature=self.txt, g_label=torch.ones(2),
            device=torch.device('cpu'),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(calls), 1)
        loss.backward()
        self.assertIsNotNone(fake.grad)


if __name__ == '__main__':
    unittest.main()
