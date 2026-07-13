import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock
import zipfile

from click.testing import CliRunner
import numpy as np
from PIL import Image
import torch

from preprocessing import preprocess_dataset
from preprocessing.split_dataset import get_all_image_files, split_dataset


class _ClipModel:
    def __init__(self, image_error=None):
        self.image_error = image_error

    def eval(self):
        return self

    def encode_image(self, images):
        if self.image_error is not None:
            raise self.image_error
        return torch.zeros(images.size(0), 512)

    def encode_text(self, tokens):
        return torch.ones(tokens.size(0), 512)


class _InputClipModel(_ClipModel):
    def encode_image(self, images):
        return images.flatten(1)[:, :512]


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @staticmethod
    def _arguments(destination, seed=42):
        return [
            "--source",
            "unused",
            "--src_data_list",
            "unused.pickle",
            "--dest",
            str(destination),
            "--emb_dim",
            "512",
            "--seed",
            str(seed),
        ]

    def _invoke_with_samples(self, destination, samples, model=None, seed=42):
        model = model or _ClipModel()
        with mock.patch.object(
            preprocess_dataset.clip, "load", return_value=(model, None)
        ), mock.patch.object(
            preprocess_dataset.clip,
            "tokenize",
            side_effect=lambda captions, truncate=True: torch.zeros(len(captions), 77, dtype=torch.long),
        ), mock.patch.object(
            preprocess_dataset, "open_dataset", return_value=(len(samples), iter(samples))
        ), mock.patch.object(
            preprocess_dataset.torch.cuda, "is_available", return_value=False
        ):
            return self.runner.invoke(
                preprocess_dataset.convert_dataset, self._arguments(destination, seed)
            )

    def test_emb_dim_is_required_and_fixed_to_512(self):
        missing = self.runner.invoke(
            preprocess_dataset.convert_dataset,
            ["--source", "x", "--src_data_list", "x", "--dest", "x.zip"],
        )
        self.assertNotEqual(missing.exit_code, 0)
        self.assertIn("--emb_dim", missing.output)
        wrong = self.runner.invoke(
            preprocess_dataset.convert_dataset,
            [
                "--source",
                "x",
                "--src_data_list",
                "x",
                "--dest",
                "x.zip",
                "--emb_dim",
                "256",
            ],
        )
        self.assertNotEqual(wrong.exit_code, 0)
        self.assertIn("512", wrong.output)

    def test_folder_reader_converts_grayscale_and_rgba_to_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "celeba-caption").mkdir()
            Image.new("L", (8, 8), 127).save(root / "images" / "gray.png")
            Image.new("RGBA", (8, 8), (1, 2, 3, 4)).save(root / "images" / "rgba.png")
            (root / "celeba-caption" / "gray.txt").write_text("gray", encoding="utf-8")
            (root / "celeba-caption" / "rgba.txt").write_text("rgba", encoding="utf-8")
            selected = root / "selected.pickle"
            with selected.open("wb") as handle:
                pickle.dump(["gray", "rgba"], handle)

            count, iterator = preprocess_dataset.open_image_folder(
                str(root), max_images=None, src_data_list=str(selected)
            )
            images = list(iterator)
            self.assertEqual(count, 2)
            self.assertEqual(len(images), 2)
            self.assertTrue(all(item["img"].shape == (8, 8, 3) for item in images))

    def test_zero_success_exits_nonzero_and_writes_no_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "failed.zip"
            result = self._invoke_with_samples(
                destination,
                [{"img": np.zeros((64, 64, 3), dtype=np.uint8), "txt": []}],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("No samples were successfully preprocessed", result.output)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).glob(".failed.zip.*.tmp")), [])

    def test_cuda_oom_is_fatal_not_a_skipped_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "oom.zip"
            error = torch.cuda.OutOfMemoryError("intentional test OOM")
            result = self._invoke_with_samples(
                destination,
                [{"img": np.zeros((64, 64, 3), dtype=np.uint8), "txt": ["caption"]}],
                model=_ClipModel(image_error=error),
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIs(result.exception, error)
            self.assertNotIn("succeeded", result.output)
            self.assertFalse(destination.exists())

    def test_success_has_rgb_png_and_matching_feature_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ok.zip"
            result = self._invoke_with_samples(
                destination,
                [{"img": np.zeros((64, 64), dtype=np.uint8), "txt": ["caption"]}],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            with zipfile.ZipFile(destination) as archive:
                metadata = json.loads(archive.read("dataset.json"))
                png_names = [name for name in archive.namelist() if name.endswith(".png")]
                self.assertEqual(len(png_names), 1)
                self.assertEqual(len(metadata["clip_img_features"]), 1)
                self.assertEqual(len(metadata["clip_txt_features"]), 1)
                self.assertEqual(len(metadata["clip_img_features"][0][1]), 512)
                self.assertEqual(metadata["preprocess"]["seed"], 42)
                self.assertEqual(metadata["preprocess"]["clip_model"], "ViT-B/32")
                with Image.open(io.BytesIO(archive.read(png_names[0]))) as image:
                    self.assertEqual(image.mode, "RGB")

    def test_nonuniform_output_attributes_are_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "mixed.zip"
            samples = [
                {"img": np.zeros((64, 64, 3), dtype=np.uint8), "txt": ["one"]},
                {"img": np.zeros((32, 32, 3), dtype=np.uint8), "txt": ["two"]},
            ]
            result = self._invoke_with_samples(destination, samples)
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("uniform attributes", result.output)
            self.assertFalse(destination.exists())

    def test_partial_failure_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset.zip"
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr("old-marker", b"preserve me")
            before = destination.read_bytes()
            samples = [
                {"img": np.zeros((64, 64, 3), dtype=np.uint8), "txt": ["valid"]},
                {"img": np.zeros((64, 64, 3), dtype=np.uint8), "txt": []},
            ]

            result = self._invoke_with_samples(destination, samples)

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Preprocessing was incomplete", result.output)
            self.assertEqual(destination.read_bytes(), before)
            self.assertEqual(list(Path(directory).glob(".dataset.zip.*.tmp")), [])

    def test_invalid_transform_does_not_open_or_replace_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset.zip"
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr("old-marker", b"preserve me")
            before = destination.read_bytes()
            arguments = self._arguments(destination) + ["--transform", "center-crop"]
            with mock.patch.object(
                preprocess_dataset.clip, "load", return_value=(_ClipModel(), None)
            ), mock.patch.object(
                preprocess_dataset, "open_dataset", return_value=(1, iter([]))
            ), mock.patch.object(
                preprocess_dataset.torch.cuda, "is_available", return_value=False
            ):
                result = self.runner.invoke(
                    preprocess_dataset.convert_dataset, arguments
                )

            self.assertNotEqual(result.exit_code, 0)
            self.assertEqual(destination.read_bytes(), before)
            self.assertEqual(list(Path(directory).glob(".dataset.zip.*.tmp")), [])

    def test_success_atomically_replaces_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset.zip"
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr("old-marker", b"replace me")

            result = self._invoke_with_samples(
                destination,
                [{"img": np.zeros((64, 64, 3), dtype=np.uint8), "txt": ["valid"]}],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            with zipfile.ZipFile(destination) as archive:
                self.assertNotIn("old-marker", archive.namelist())
                self.assertIn("dataset.json", archive.namelist())

    def test_fixed_seed_produces_byte_stable_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            pixels = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
            samples = [{"img": pixels, "txt": ["caption"]}]
            first = Path(directory) / "first.zip"
            second = Path(directory) / "second.zip"
            different = Path(directory) / "different.zip"

            result_a = self._invoke_with_samples(first, samples, _InputClipModel(), seed=42)
            result_b = self._invoke_with_samples(second, samples, _InputClipModel(), seed=42)
            result_c = self._invoke_with_samples(different, samples, _InputClipModel(), seed=43)

            self.assertEqual(result_a.exit_code, 0, result_a.output)
            self.assertEqual(result_b.exit_code, 0, result_b.output)
            self.assertEqual(result_c.exit_code, 0, result_c.output)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotEqual(first.read_bytes(), different.read_bytes())
            with zipfile.ZipFile(first) as archive:
                for info in archive.infolist():
                    self.assertEqual(info.date_time, preprocess_dataset.ZIP_ENTRY_TIMESTAMP)
                    self.assertEqual((info.external_attr >> 16) & 0o777, 0o644)

    def test_caption_limit_counts_nonempty_captions(self):
        captions = ["  "] * 10 + ["caption eleven"]
        with mock.patch.object(
            preprocess_dataset.clip,
            "tokenize",
            side_effect=lambda values, truncate=True: torch.zeros(
                len(values), 77, dtype=torch.long
            ),
        ):
            features = preprocess_dataset.encode_text_features(
                _ClipModel(), captions, "cpu"
            )
        self.assertEqual(len(features), 1)

    def test_folder_reader_supports_multi_dot_stems(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "celeba-caption").mkdir()
            Image.new("RGB", (8, 8)).save(root / "images" / "person.v1.jpg")
            (root / "celeba-caption" / "person.v1.txt").write_text(
                "caption", encoding="utf-8"
            )
            selected = root / "selected.pickle"
            with selected.open("wb") as handle:
                pickle.dump(["person.v1"], handle)

            count, iterator = preprocess_dataset.open_image_folder(
                str(root), max_images=None, src_data_list=str(selected)
            )

            self.assertEqual(count, 1)
            self.assertEqual(list(iterator)[0]["txt"], ["caption"])

    def test_source_reader_rejects_missing_selected_stem(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "celeba-caption").mkdir()
            Image.new("RGB", (8, 8)).save(root / "images" / "present.jpg")
            selected = root / "selected.pickle"
            with selected.open("wb") as handle:
                pickle.dump(["present", "missing"], handle)

            with self.assertRaisesRegex(ValueError, "missing from") as error:
                preprocess_dataset.open_image_folder(
                    str(root), max_images=None, src_data_list=str(selected)
                )

            self.assertIn("missing", str(error.exception))

    def test_source_readers_reject_duplicate_stems_with_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "nested").mkdir(parents=True)
            (root / "celeba-caption").mkdir()
            Image.new("RGB", (8, 8)).save(root / "images" / "same.jpg")
            Image.new("RGB", (8, 8)).save(root / "images" / "nested" / "same.png")
            selected = root / "selected.pickle"
            with selected.open("wb") as handle:
                pickle.dump(["same"], handle)
            with self.assertRaisesRegex(ValueError, "duplicate image stems") as error:
                preprocess_dataset.open_image_folder(
                    str(root), max_images=None, src_data_list=str(selected)
                )
            self.assertIn("same.jpg", str(error.exception))
            self.assertIn("nested/same.png", str(error.exception))

            zip_root = root / "zip-source"
            zip_root.mkdir()
            with zipfile.ZipFile(zip_root / "image.zip", "w") as archive:
                archive.writestr("images/same.jpg", b"one")
                archive.writestr("other/same.png", b"two")
            with zipfile.ZipFile(zip_root / "text.zip", "w") as archive:
                archive.writestr("celeba-caption/same.txt", "caption")
            with self.assertRaisesRegex(ValueError, "duplicate image stems") as error:
                preprocess_dataset.open_image_zip(
                    str(zip_root), max_images=None, src_data_list=str(selected)
                )
            self.assertIn("images/same.jpg", str(error.exception))
            self.assertIn("other/same.png", str(error.exception))


class SplitDatasetTests(unittest.TestCase):
    def test_empty_source_and_invalid_ranges_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                split_dataset(directory, train_ratio=1.0)
            with self.assertRaisesRegex(ValueError, "max_images must be positive"):
                split_dataset(directory, max_images=0)
            with self.assertRaisesRegex(ValueError, "no input images"):
                split_dataset(directory)

    def test_duplicate_stems_are_rejected_before_split(self):
        with tempfile.TemporaryDirectory() as directory:
            image_zip = Path(directory) / "image.zip"
            with zipfile.ZipFile(image_zip, "w") as archive:
                archive.writestr("images/duplicate.jpg", b"one")
                archive.writestr("alternate/duplicate.png", b"two")

            with self.assertRaisesRegex(ValueError, "duplicate image stems") as error:
                get_all_image_files(directory)

            self.assertIn("images/duplicate.jpg", str(error.exception))
            self.assertIn("alternate/duplicate.png", str(error.exception))


if __name__ == "__main__":
    unittest.main()
