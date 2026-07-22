import math
import unittest

import torch

from config.config import CLIPConfig
from criteria.loss import D_loss, G_loss, contrastive_loss_D, conditioning_gate
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
            # A fixed, non-0.5 logit (independent of the actual condition
            # values) so tests can pin an exact loss value rather than only
            # checking that two equal-probability terms average out.
            logits = torch.full_like(condition.sum(dim=1), 2.0)
        else:
            logits = img.flatten(1).sum(dim=1) * 0.0
        output = torch.sigmoid(logits)
        if not return_details:
            return output, None

        details = {'conditional': output}
        if mismatched_condition is not None:
            self.conditions.append(mismatched_condition.detach().clone())
            self.alignment_requests.append(False)
            # Distinct from the matched-condition logit above: this lets a
            # test pin the exact 0.5/0.5 fake/mismatched rescale weighting in
            # criteria/loss.py instead of only proving the two weights sum
            # to 1.0 (which a p=0.5-for-everything D cannot distinguish from
            # a wrong e.g. 0.3/0.7 split).
            details['mismatched_conditional'] = torch.sigmoid(
                torch.full_like(mismatched_condition.sum(dim=1), -1.0)
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
        bce = torch.nn.BCELoss()

        loss = D_loss(
            real, fake, model, bce,
            use_uncond_loss=False, use_contrastive_loss=False,
            gamma=1.0, mu=mu, txt_feature=mu,
            d_fake_label=labels_fake, d_real_label=labels_real,
        )

        self.assertEqual(loss.ndim, 0)
        self.assertEqual(len(model.conditions), 3)
        self.assertEqual(model.forward_calls, 2)
        torch.testing.assert_close(model.conditions[-1], mu.roll(1, 0))
        self.assertEqual(model.alignment_requests, [False, False, False])

        # The recording D emits DISTINCT, non-0.5 logits for the
        # matched-condition head (2.0, used by both fake and real matched
        # pairings) and the mismatched-condition head (-1.0). This pins the
        # exact 0.5/0.5 fake/mismatched-real rescale weighting in
        # criteria/loss.py: an incorrect split (e.g. 0.3/0.7) would NOT
        # reproduce `expected` below, whereas the old p=0.5-for-everything
        # recorder made every split summing to 1.0 indistinguishable.
        p_matched = torch.sigmoid(torch.tensor(2.0)).expand(batch_size)
        p_mismatched = torch.sigmoid(torch.tensor(-1.0)).expand(batch_size)
        expected = (
            bce(p_matched, labels_real)
            + 0.5 * bce(p_matched, labels_fake)
            + 0.5 * bce(p_mismatched, labels_fake)
        )
        torch.testing.assert_close(loss, expected)

        # Confirm the pinning is non-trivial: a wrong 0.3/0.7 split would have
        # produced a different total given these distinct logits.
        wrong_split = (
            bce(p_matched, labels_real)
            + 0.3 * bce(p_matched, labels_fake)
            + 0.7 * bce(p_mismatched, labels_fake)
        )
        self.assertFalse(torch.allclose(loss, wrong_split))

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

    # --- --cond_warmup_epochs / --cond_ramp_epochs staging (cond_gate) ---

    def test_cond_gate_default_reproduces_original_mismatched_balance(self):
        """cond_gate's default (1.0, i.e. --cond_warmup_epochs 0) must reproduce the
        original hardcoded 0.5/0.5 fake/mismatched split bit-for-bit, whether cond_gate
        is passed explicitly or omitted."""
        model = _RecordingDiscriminator()
        batch_size = 3
        real = torch.randn(batch_size, 3, 4, 4)
        fake = torch.randn_like(real)
        mu = torch.arange(batch_size * 2, dtype=torch.float32).view(batch_size, 2)
        labels_real = torch.ones(batch_size)
        labels_fake = torch.zeros(batch_size)
        bce = torch.nn.BCELoss()

        kwargs = dict(
            use_uncond_loss=False, use_contrastive_loss=False,
            gamma=1.0, mu=mu, txt_feature=mu,
            d_fake_label=labels_fake, d_real_label=labels_real,
        )
        loss_omitted = D_loss(real, fake, model, bce, **kwargs)
        loss_explicit = D_loss(real, fake, model, bce, cond_gate=1.0, **kwargs)

        p_matched = torch.sigmoid(torch.tensor(2.0)).expand(batch_size)
        p_mismatched = torch.sigmoid(torch.tensor(-1.0)).expand(batch_size)
        expected = (
            bce(p_matched, labels_real)
            + 0.5 * bce(p_matched, labels_fake)
            + 0.5 * bce(p_mismatched, labels_fake)
        )
        torch.testing.assert_close(loss_omitted, expected)
        torch.testing.assert_close(loss_explicit, expected)

    def test_cond_gate_zero_disables_gated_terms_without_halving_fake_cond(self):
        """During warm-up (cond_gate=0.0) the gamma alignment terms and the
        mismatched-condition negative must be fully absent (D is not even asked to
        compute them), AND d_loss_fake_cond must NOT still carry the 0.5 mismatched
        rebalancing weight -- that would silently halve the plain real/fake signal
        that is supposed to always train from epoch 0."""
        model = _RecordingDiscriminator()
        batch_size = 2
        real = torch.randn(batch_size, 3, 4, 4)
        fake = torch.randn_like(real)
        mu = torch.randn(batch_size, 2)
        labels_real = torch.ones(batch_size)
        labels_fake = torch.zeros(batch_size)
        bce = torch.nn.BCELoss()

        loss = D_loss(
            real, fake, model, bce,
            use_uncond_loss=False, use_contrastive_loss=True,
            gamma=5.0, mu=mu, txt_feature=mu,
            d_fake_label=labels_fake, d_real_label=labels_real,
            use_mismatched_condition=True,
            cond_gate=0.0,
        )

        # Neither the mismatched negative nor the alignment head were requested at all.
        self.assertEqual(model.forward_calls, 2)
        self.assertEqual(len(model.conditions), 2)
        self.assertEqual(model.alignment_requests, [False, False])

        # The plain real/fake BCE is present at FULL (unhalved) weight and non-zero.
        p_matched = torch.sigmoid(torch.tensor(2.0)).expand(batch_size)
        expected = bce(p_matched, labels_real) + bce(p_matched, labels_fake)
        torch.testing.assert_close(loss, expected)
        self.assertGreater(loss.item(), 0.0)

    def test_cond_gate_scales_gamma_alignment_terms(self):
        """--gamma must actually scale the D-side alignment InfoNCE terms: gamma=0
        must zero them exactly, and increasing gamma must scale their contribution
        linearly (contrastive_loss_D itself does not depend on gamma)."""
        model = _RecordingDiscriminator()
        batch_size = 2
        real = torch.randn(batch_size, 3, 4, 4)
        fake = torch.randn_like(real)
        mu = torch.randn(batch_size, 2)
        labels_real = torch.ones(batch_size)
        labels_fake = torch.zeros(batch_size)
        bce = torch.nn.BCELoss()

        def run(gamma):
            return D_loss(
                real, fake, model, bce,
                use_uncond_loss=False, use_contrastive_loss=True,
                gamma=gamma, mu=mu, txt_feature=mu,
                d_fake_label=labels_fake, d_real_label=labels_real,
                use_mismatched_condition=False,
                cond_gate=1.0,
            )

        loss_gamma0 = run(0.0)
        loss_gamma2 = run(2.0)
        loss_gamma4 = run(4.0)

        p_matched = torch.sigmoid(torch.tensor(2.0)).expand(batch_size)
        base = bce(p_matched, labels_real) + bce(p_matched, labels_fake)
        torch.testing.assert_close(loss_gamma0, base)

        # _RecordingDiscriminator's 'alignment' output is zeros_like(condition), so
        # contrastive_loss_D(zeros, mu) is the fixed-point cross-entropy of an
        # all-zero (batch_size x batch_size) logit matrix: ln(batch_size).
        per_term = math.log(batch_size)
        expected_gamma2 = base + 2.0 * (2 * per_term)
        expected_gamma4 = base + 4.0 * (2 * per_term)
        torch.testing.assert_close(loss_gamma2, expected_gamma2)
        torch.testing.assert_close(loss_gamma4, expected_gamma4)

        # Linear in gamma: doubling gamma doubles the added contrastive contribution.
        delta_2 = loss_gamma2 - loss_gamma0
        delta_4 = loss_gamma4 - loss_gamma0
        torch.testing.assert_close(delta_4, 2.0 * delta_2)
        self.assertGreater(delta_2.item(), 0.0)

    def test_cond_gate_ramp_preserves_1to1_mass_balance(self):
        """A partial ramp value (e.g. from --cond_ramp_epochs) must still split the
        fake-cond / mismatched-cond negative mass so it sums to exactly 1 (matching
        the real-cond positive weight of 1), interpolating between 'no mismatched
        term, full fake_cond weight' (gate=0) and the original 0.5/0.5 (gate=1)."""
        model = _RecordingDiscriminator()
        batch_size = 3
        real = torch.randn(batch_size, 3, 4, 4)
        fake = torch.randn_like(real)
        mu = torch.arange(batch_size * 2, dtype=torch.float32).view(batch_size, 2)
        labels_real = torch.ones(batch_size)
        labels_fake = torch.zeros(batch_size)
        bce = torch.nn.BCELoss()

        cond_gate = 0.5
        loss = D_loss(
            real, fake, model, bce,
            use_uncond_loss=False, use_contrastive_loss=False,
            gamma=1.0, mu=mu, txt_feature=mu,
            d_fake_label=labels_fake, d_real_label=labels_real,
            use_mismatched_condition=True,
            cond_gate=cond_gate,
        )

        p_matched = torch.sigmoid(torch.tensor(2.0)).expand(batch_size)
        p_mismatched = torch.sigmoid(torch.tensor(-1.0)).expand(batch_size)
        fake_cond_weight = 1.0 - 0.5 * cond_gate
        mismatched_weight = 0.5 * cond_gate
        self.assertAlmostEqual(fake_cond_weight + mismatched_weight, 1.0)
        expected = (
            bce(p_matched, labels_real)
            + fake_cond_weight * bce(p_matched, labels_fake)
            + mismatched_weight * bce(p_mismatched, labels_fake)
        )
        torch.testing.assert_close(loss, expected)


class ConditioningGateTests(unittest.TestCase):
    def test_defaults_are_always_fully_on(self):
        for epoch in (0, 1, 5, 100):
            self.assertEqual(conditioning_gate(epoch, warmup_epochs=0, ramp_epochs=0), 1.0)

    def test_hard_switch_at_warmup_boundary(self):
        for epoch in range(5):
            self.assertEqual(conditioning_gate(epoch, warmup_epochs=5, ramp_epochs=0), 0.0)
        for epoch in range(5, 10):
            self.assertEqual(conditioning_gate(epoch, warmup_epochs=5, ramp_epochs=0), 1.0)

    def test_linear_ramp_after_warmup(self):
        warmup_epochs, ramp_epochs = 5, 4
        for epoch in range(warmup_epochs):
            self.assertEqual(conditioning_gate(epoch, warmup_epochs, ramp_epochs), 0.0)
        expected = [0.25, 0.5, 0.75, 1.0]
        for offset, value in enumerate(expected):
            epoch = warmup_epochs + offset
            self.assertAlmostEqual(
                conditioning_gate(epoch, warmup_epochs, ramp_epochs), value
            )
        # Stays clamped at 1.0 well past the ramp.
        self.assertEqual(conditioning_gate(warmup_epochs + 50, warmup_epochs, ramp_epochs), 1.0)


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

    def _alignment_head_counter(self):
        calls = []
        handle = self.model.align_cond_discriminator.register_forward_hook(
            lambda *_args: calls.append(1)
        )
        self.addCleanup(handle.remove)
        return calls

    def test_gamma_zero_skips_alignment_head_forward_entirely(self):
        """--gamma 0 must skip the alignment head's forward (and thus its
        BatchNorm running-stat updates) rather than compute it and multiply the
        result by zero afterward. Spies on the real align_cond_discriminator
        module -- not just the zeroed loss value -- so a regression that still
        runs the forward but zeroes the contribution would be caught."""
        calls = self._alignment_head_counter()
        loss = D_loss(
            self.real, self.fake, self.model, self.loss_fn,
            use_uncond_loss=False, use_contrastive_loss=True,
            gamma=0.0, mu=self.mu, txt_feature=self.txt,
            d_fake_label=torch.zeros(2), d_real_label=torch.ones(2),
            use_mismatched_condition=False,
            cond_gate=1.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(calls), 0)

    def test_gamma_default_still_calls_alignment_head_forward(self):
        """Companion to the gamma=0 skip test above: at the default (non-zero)
        gamma weight, the alignment head must actually be invoked (once per
        D_loss forward call, i.e. fake + real), proving the gamma=0 case above
        is a real skip and not a coincidental zero-call setup."""
        calls = self._alignment_head_counter()
        loss = D_loss(
            self.real, self.fake, self.model, self.loss_fn,
            use_uncond_loss=False, use_contrastive_loss=True,
            gamma=5.0, mu=self.mu, txt_feature=self.txt,
            d_fake_label=torch.zeros(2), d_real_label=torch.ones(2),
            use_mismatched_condition=False,
            cond_gate=1.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(calls), 2)

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


class _ZeroImageClipStub:
    """Deterministic clip_model stand-in: encode_image always returns zeros,
    independent of image content. contrastive_loss_G then collapses to the fixed
    cross-entropy of an all-zero (batch_size x batch_size) logit matrix, i.e.
    ln(batch_size) -- isolating --lam's linear scaling from any real CLIP/image
    dependence (and avoiding a network-dependent CLIP weight load in a unit test).
    Also records how many times encode_image was called, so tests can assert the
    CLIP forward itself was skipped rather than merely zeroed afterward.
    """
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.encode_image_calls = 0

    def encode_image(self, images):
        self.encode_image_calls += 1
        return torch.zeros(images.shape[0], self.feature_dim)


class LamScalingTests(unittest.TestCase):
    def test_lam_scales_the_256px_clip_contrastive_term(self):
        """--lam must actually scale the G-side CLIP contrastive term, which only
        activates at the >= CLIPConfig.MIN_QUALITY_SIZE stage (loss.py G_loss)."""
        model = _RecordingDiscriminator()
        batch_size = 2
        feature_dim = 2
        size = CLIPConfig.MIN_QUALITY_SIZE
        real = torch.randn(batch_size, 3, size, size)
        fake = torch.randn_like(real).requires_grad_(True)
        mu = torch.randn(batch_size, feature_dim)
        clip_stub = _ZeroImageClipStub(feature_dim)

        def run(lam):
            # gamma=0.0 zeroes G_loss's other (D-alignment) contrastive term so the
            # lam-scaled CLIP term is isolated.
            return G_loss(
                real, fake, model, torch.nn.BCELoss(),
                use_uncond_loss=False, use_contrastive_loss=True, use_mixed_loss=False,
                clip_model=clip_stub, gamma=0.0, lam=lam,
                mu=mu, txt_feature=mu, g_label=torch.ones(batch_size),
                device=torch.device('cpu'),
            )

        loss_lam0 = run(0.0)
        loss_lam3 = run(3.0)

        per_term = math.log(batch_size)
        delta = loss_lam3 - loss_lam0
        torch.testing.assert_close(delta, torch.tensor(3.0 * per_term))
        self.assertGreater(delta.item(), 0.0)

    def test_lam_zero_skips_clip_forward_entirely(self):
        """--lam 0 must skip contrastive_loss_G's CLIP forward (encode_image)
        entirely rather than compute it and multiply the result by zero
        afterward -- otherwise the flag does not shed the CLIP memory/compute
        cost its help text implies. Spies on encode_image's call count, not
        just the zeroed loss value, so a regression that still runs the CLIP
        forward but zeroes the contribution would be caught."""
        model = _RecordingDiscriminator()
        batch_size = 2
        feature_dim = 2
        size = CLIPConfig.MIN_QUALITY_SIZE
        real = torch.randn(batch_size, 3, size, size)
        fake = torch.randn_like(real).requires_grad_(True)
        mu = torch.randn(batch_size, feature_dim)

        clip_stub_zero = _ZeroImageClipStub(feature_dim)
        G_loss(
            real, fake, model, torch.nn.BCELoss(),
            use_uncond_loss=False, use_contrastive_loss=True, use_mixed_loss=False,
            clip_model=clip_stub_zero, gamma=0.0, lam=0.0,
            mu=mu, txt_feature=mu, g_label=torch.ones(batch_size),
            device=torch.device('cpu'),
        )
        self.assertEqual(clip_stub_zero.encode_image_calls, 0)

        clip_stub_default = _ZeroImageClipStub(feature_dim)
        G_loss(
            real, fake, model, torch.nn.BCELoss(),
            use_uncond_loss=False, use_contrastive_loss=True, use_mixed_loss=False,
            clip_model=clip_stub_default, gamma=0.0, lam=10.0,
            mu=mu, txt_feature=mu, g_label=torch.ones(batch_size),
            device=torch.device('cpu'),
        )
        self.assertGreater(clip_stub_default.encode_image_calls, 0)


if __name__ == '__main__':
    unittest.main()
