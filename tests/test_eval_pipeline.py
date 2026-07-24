import hashlib
import csv
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import math
import torch

from criteria.metric import calculate_clip_score
from experiments import eval_curve, plot_compare
from scripts import checkpoint_config
from utils.utils import save_metrics_to_csv


class _Metric:
    def to(self, _device):
        return self

    def update(self, _images, real=None):
        return None

    def compute(self):
        return torch.tensor(1.25)


class _InceptionMetric(_Metric):
    def compute(self):
        return torch.tensor(2.5), torch.tensor(0.25)


class _ClipModel:
    """Minimal stand-in for a loaded CLIP model: only encode_image() is exercised."""

    def eval(self):
        return self

    def encode_image(self, images):
        return torch.zeros(images.shape[0], 512)


class _Generator:
    def __init__(self, generated):
        self.generated = generated

    def eval(self):
        return self

    def __call__(self, captions, noise):
        # Simulate the generator's independent conditioning-augmentation draw.
        conditioning_noise = torch.randn(captions.size(0), 2)
        self.generated.append((noise.detach().clone(), conditioning_noise))
        images = torch.zeros(captions.size(0), 3, 8, 8)
        return [images], None, None


class EvalCurveTests(unittest.TestCase):
    def test_cli_preserves_four_positionals_and_accepts_optional_seed(self):
        args = eval_curve.parse_args(["test.zip", "ckpt", "auto", "out.json"])
        self.assertEqual(args.seed, 42)
        args = eval_curve.parse_args(
            ["test.zip", "ckpt", "1,2", "out.json", "--seed", "7"]
        )
        self.assertEqual(args.seed, 7)
        args = eval_curve.parse_args(["test.zip", "ckpt", "1", "out.json", "9"])
        self.assertEqual(args.seed, 9)

    def test_each_checkpoint_reseeds_after_loading_and_is_order_independent(self):
        real = torch.zeros(2, 3, 8, 8)
        captions = torch.zeros(2, 512)
        config = dict(eval_curve.LEGACY_MODEL_CONFIG)
        config["noise_dim"] = 4
        generated = []
        events = []

        def fake_load(*_args, **_kwargs):
            events.append("load")
            torch.rand(17)  # checkpoint loading is allowed to disturb global RNG

        def fake_seed(seed):
            events.append(("seed", seed))
            torch.manual_seed(seed)

        with mock.patch.object(
            eval_curve, "make_generator", side_effect=lambda *_: _Generator(generated)
        ), mock.patch.object(eval_curve, "load_checkpoint", side_effect=fake_load), mock.patch.object(
            eval_curve, "seed_fix", side_effect=fake_seed
        ), mock.patch.object(
            eval_curve, "FrechetInceptionDistance", return_value=_Metric()
        ), mock.patch.object(
            eval_curve, "InceptionScore", return_value=_InceptionMetric()
        ), mock.patch.object(
            eval_curve.torch.cuda, "is_available", return_value=False
        ):
            eval_curve.evaluate_checkpoint(
                Path("unused"), 10, real, captions, config, 123, "cpu", _ClipModel()
            )
            torch.rand(31)
            eval_curve.evaluate_checkpoint(
                Path("unused"), 20, real, captions, config, 123, "cpu", _ClipModel()
            )

        self.assertEqual(events, ["load", ("seed", 123), "load", ("seed", 123)])
        torch.testing.assert_close(generated[0][0], generated[1][0])
        torch.testing.assert_close(generated[0][1], generated[1][1])

    def test_no_checkpoint_is_a_nonzero_cli_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "eval.json"
            exit_code = eval_curve.main(
                [
                    str(Path(directory) / "missing.zip"),
                    directory,
                    "auto",
                    str(output),
                ]
            )
            self.assertNotEqual(exit_code, 0)
            self.assertFalse(output.exists())
            self.assertFalse(Path(f"{output}.provenance.json").exists())

    def test_explicit_epoch_list_fails_with_every_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "epoch_0_Gen.pt").write_bytes(b"zero")
            (root / "epoch_20_Gen.pt").write_bytes(b"twenty")

            with self.assertRaises(FileNotFoundError) as raised:
                eval_curve.existing_checkpoints("0,10,20,30", root)

            message = str(raised.exception)
            self.assertIn("explicitly requested", message)
            self.assertIn("10, 30", message)
            self.assertNotIn("epoch 0", message)
            self.assertNotIn("epoch 20", message)

    def test_result_schema_stays_flat_and_provenance_is_a_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "epoch_3_Gen.pt"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "eval.json"
            args = types.SimpleNamespace(
                test_zip=str(root / "test.zip"),
                checkpoint_dir=str(root),
                epochs="3",
                output=str(output),
                seed=11,
            )
            config = dict(eval_curve.LEGACY_MODEL_CONFIG)
            metrics = {
                "fid": 4.0,
                "is_mean": 2.0,
                "is_std": 0.1,
                "clip_score": 0.25,
                "clip_diversity": 0.5,
            }
            provenance = {"schema_version": 1, "seed": 11}
            with mock.patch.object(
                eval_curve, "existing_checkpoints", return_value=[(3, checkpoint)]
            ), mock.patch.object(
                eval_curve,
                "load_test_data",
                return_value=(
                    torch.zeros(1, 3, 8, 8),
                    torch.zeros(1, 512),
                    ["image.png"],
                ),
            ), mock.patch.object(
                eval_curve,
                "read_checkpoint_config",
                return_value=(config, "legacy_inferred", {"epoch": 3}),
            ), mock.patch.object(
                eval_curve, "evaluate_checkpoint", return_value=metrics
            ), mock.patch.object(
                eval_curve, "build_provenance", return_value=provenance
            ), mock.patch.object(
                eval_curve.torch.cuda, "is_available", return_value=False
            ), mock.patch.object(
                eval_curve.CLIPConfig, "load_clip", return_value=(_ClipModel(), None)
            ):
                result_path, provenance_path = eval_curve.run(args)

            self.assertEqual(json.loads(result_path.read_text()), {"3": metrics})
            self.assertEqual(provenance_path, Path(f"{output}.provenance.json"))
            self.assertEqual(json.loads(provenance_path.read_text()), provenance)
            self.assertEqual(
                provenance["result_artifact"]["sha256"],
                hashlib.sha256(result_path.read_bytes()).hexdigest(),
            )

    def test_provenance_contains_required_hashes_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "test.zip"
            checkpoint = root / "epoch_1_Gen.pt"
            data.write_bytes(b"data")
            checkpoint.write_bytes(b"checkpoint")
            details = {
                "1": {
                    "model_config": dict(eval_curve.LEGACY_MODEL_CONFIG),
                    "model_config_source": "legacy_inferred",
                    "checkpoint_epoch": 1,
                }
            }
            payload = eval_curve.build_provenance(
                test_zip=data,
                sample_count=5,
                caption_dimension=512,
                checkpoints=[(1, checkpoint)],
                checkpoint_details=details,
                seed=42,
                device="cpu",
                epoch_arg="1",
            )
            self.assertEqual(payload["sample_count"], 5)
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(
                payload["data"]["sha256"], hashlib.sha256(b"data").hexdigest()
            )
            self.assertEqual(
                payload["checkpoints"]["1"]["sha256"],
                hashlib.sha256(b"checkpoint").hexdigest(),
            )
            self.assertIn("git_commit", payload)
            self.assertIn("torch", payload["package_versions"])
            self.assertEqual(payload["config"]["caption_embedding_dimension"], 512)


class StandaloneCheckpointConfigTests(unittest.TestCase):
    def test_v2_config_overrides_cli_before_model_construction(self):
        model_config = dict(checkpoint_config.LEGACY_MODEL_CONFIG)
        model_config.update(
            {
                "noise_dim": 77,
                "condition_dim": 64,
                "conditioning_activation": "linear",
                "alignment_mode": "image_only",
            }
        )
        metadata = {
            "format_version": 2,
            "legacy": False,
            "model_config": model_config,
        }
        args = types.SimpleNamespace(
            checkpoint_path="ckpt",
            load_epoch=4,
            noise_dim=999,
        )
        with mock.patch.object(
            checkpoint_config, "peek_checkpoint_metadata", return_value=metadata
        ):
            returned, resolved = checkpoint_config.apply_checkpoint_model_config(args)
        self.assertIs(returned["model_config"], model_config)
        self.assertEqual(resolved["noise_dim"], 77)
        self.assertEqual(args.noise_dim, 77)
        self.assertEqual(args.conditioning_activation, "linear")

    def test_metrics_csv_records_checkpoint_identity_and_rejects_legacy_header(self):
        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(
                result_path=directory,
                eval_checkpoint_path="/tmp/epoch_3_Gen.pt",
                eval_checkpoint_sha256="abc123",
                eval_checkpoint_format=2,
                eval_checkpoint_legacy=False,
                eval_generator_weight_kind="raw",
                eval_conditioning_activation="linear",
                eval_alignment_mode="image_only",
                load_epoch=3,
                seed=42,
            )
            metrics = {
                "clip_score": 0.1,
                "fid_score": 4.2,
                "inception_score_mean": 2.0,
                "inception_score_std": 0.2,
                "samples_processed": 8,
            }
            save_metrics_to_csv(args, metrics)
            path = Path(directory) / "metrics.csv"
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["checkpoint_sha256"], "abc123")
            self.assertEqual(rows[0]["eval_seed"], "42")

            path.write_text("epoch,fid_score\n3,4.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompatible schema"):
                save_metrics_to_csv(args, metrics)


class PlotComparisonContractTests(unittest.TestCase):
    @staticmethod
    def _provenance(use_diffaugment):
        training_config = {
            "seed": 42,
            "use_diffaugment": use_diffaugment,
            "diffaugment_policy": (
                "color,translation,cutout" if use_diffaugment else ""
            ),
            "batch_size": 4,
            "target_num_epochs": 100,
        }
        checkpoint = {
            "model_config": dict(eval_curve.LEGACY_MODEL_CONFIG),
            "training_config": training_config,
            "schedule_config": {
                "phase_start_epoch": 0,
                "phase_end_epoch": 100,
                "t_max": 100,
                "scheduler_type": "CosineAnnealingLR",
            },
            "training_provenance": {
                "dataset": {"sha256": "train-data", "size_bytes": 10},
                "source": {"sha256": "train-code"},
                "runtime": {"torch": "2.4.0"},
                "hardware": {"gpu": "test"},
            },
        }
        return {
            "git_commit": "commit",
            "git_dirty": False,
            "seed": 42,
            "sample_count": 8,
            "caption_selection": "first_stored_caption_per_image",
            "data": {"sha256": "test-data"},
            "hardware": {"gpu": "test"},
            "package_versions": {"torch": "2.4.0"},
            "config": {
                "caption_embedding_dimension": 512,
                "fid_feature_dimension": 2048,
                "inception_input": "uint8_rgb",
                "batch_size": 32,
                "evaluated_epochs": [20],
            },
            "checkpoints": {"20": checkpoint},
        }

    def test_strict_comparison_allows_only_declared_training_difference(self):
        baseline = self._provenance(False)
        diffaugment = self._provenance(True)
        plot_compare.validate_comparable(
            [baseline, diffaugment],
            ["use_diffaugment", "diffaugment_policy"],
        )

        diffaugment["checkpoints"]["20"]["training_provenance"]["source"][
            "sha256"
        ] = "different-code"
        with self.assertRaisesRegex(ValueError, "training data/source"):
            plot_compare.validate_comparable(
                [baseline, diffaugment],
                ["use_diffaugment", "diffaugment_policy"],
            )

    def test_strict_comparison_rejects_dirty_evaluation(self):
        baseline = self._provenance(False)
        diffaugment = self._provenance(True)
        diffaugment["git_dirty"] = True
        with self.assertRaisesRegex(ValueError, "dirty or unknown Git state"):
            plot_compare.validate_comparable(
                [baseline, diffaugment],
                ["use_diffaugment", "diffaugment_policy"],
            )

    def test_diffaugment_pair_validates_direction_and_policy(self):
        baseline = self._provenance(False)
        diffaugment = self._provenance(True)
        plot_compare.validate_diffaugment_pair(
            [baseline, diffaugment], "color, translation, cutout"
        )

        with self.assertRaisesRegex(ValueError, "baseline"):
            plot_compare.validate_diffaugment_pair(
                [diffaugment, baseline], "color,translation,cutout"
            )

        with self.assertRaisesRegex(ValueError, "policy mismatch"):
            plot_compare.validate_diffaugment_pair(
                [baseline, diffaugment], "translation,cutout"
            )

    def test_strict_provenance_binds_exact_result_json_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "eval.json"
            result_path.write_text('{"20":{"fid":1.0}}\n', encoding="utf-8")
            provenance = self._provenance(False)
            provenance["result_artifact"] = {
                "path": str(result_path),
                "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            }
            Path(f"{result_path}.provenance.json").write_text(
                json.dumps(provenance), encoding="utf-8"
            )
            self.assertEqual(
                plot_compare.load_provenance(result_path)["result_artifact"]["sha256"],
                provenance["result_artifact"]["sha256"],
            )

            result_path.write_text('{"20":{"fid":999.0}}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                plot_compare.load_provenance(result_path)


class MeanPairwiseCosineDistanceTests(unittest.TestCase):
    """Hand-computed references for eval_curve._mean_pairwise_cosine_distance.

    Unlike the _ClipModel stub above (all-zero features -> the formula collapses
    to a fixed value regardless of whether the pair-count divisor or the 1-cos
    conversion is right), these pin down the closed form against inputs whose
    correct answer can be checked by hand.
    """

    def test_three_mutually_orthogonal_unit_vectors_give_distance_one(self):
        features = torch.eye(3)  # e1, e2, e3: every pairwise cosine similarity is 0
        distance = eval_curve._mean_pairwise_cosine_distance(features)
        self.assertAlmostEqual(distance, 1.0, places=6)

    def test_two_identical_unit_vectors_give_distance_zero(self):
        features = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        distance = eval_curve._mean_pairwise_cosine_distance(features)
        self.assertAlmostEqual(distance, 0.0, places=6)

    def test_two_opposite_unit_vectors_give_distance_two(self):
        features = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        distance = eval_curve._mean_pairwise_cosine_distance(features)
        self.assertAlmostEqual(distance, 2.0, places=6)

    def test_fewer_than_two_features_gives_nan(self):
        self.assertTrue(math.isnan(eval_curve._mean_pairwise_cosine_distance(torch.zeros(1, 4))))
        self.assertTrue(math.isnan(eval_curve._mean_pairwise_cosine_distance(torch.zeros(0, 4))))


class _NonDegenerateClipModel:
    """CLIP stand-in that returns distinct, non-zero, non-unit-norm image features.

    Unlike the all-zero _ClipModel stub used elsewhere in this file, this lets a
    test pin the exact output of calculate_clip_score's normalize + dot-product
    math instead of a degenerate value every formula produces.
    """

    def __init__(self, raw_image_features: torch.Tensor):
        self._raw_image_features = raw_image_features

    def eval(self):
        return self

    def encode_image(self, images):
        assert images.shape[0] == self._raw_image_features.shape[0]
        return self._raw_image_features


class CalculateClipScoreMathTests(unittest.TestCase):
    def test_return_features_matches_hand_computed_cosine_and_is_unit_norm(self):
        # Raw (pre-normalization) CLIP image features chosen so unit-norm is not
        # already 1.0 -- this exercises the normalize() call, not just the dot
        # product.
        raw_image_features = torch.tensor([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        # unit_image_features = [[0.6, 0.8, 0, 0], [0, 0, 1, 0]]
        text_features = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        images = torch.zeros(2, 3, 8, 8)

        clip_model = _NonDegenerateClipModel(raw_image_features)
        score, features = calculate_clip_score(
            images, text_features, clip_model, return_features=True
        )

        # Hand-computed: row0 dot(unit_img0, text0) = 0.6*1 + 0.8*0 = 0.6
        #                row1 dot(unit_img1, text1) = 0*0 + 0*1 + 1*0 + 0*0 = 0.0
        # mean = (0.6 + 0.0) / 2 = 0.3
        self.assertAlmostEqual(score, 0.3, places=6)

        expected_features = torch.tensor([[0.6, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        self.assertTrue(torch.allclose(features, expected_features, atol=1e-6))
        self.assertTrue(
            torch.allclose(features.norm(dim=-1), torch.ones(2), atol=1e-6)
        )

    def test_return_features_false_returns_only_the_score(self):
        raw_image_features = torch.tensor([[1.0, 0.0]])
        text_features = torch.tensor([[1.0, 0.0]])
        images = torch.zeros(1, 3, 8, 8)

        clip_model = _NonDegenerateClipModel(raw_image_features)
        result = calculate_clip_score(images, text_features, clip_model)

        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
