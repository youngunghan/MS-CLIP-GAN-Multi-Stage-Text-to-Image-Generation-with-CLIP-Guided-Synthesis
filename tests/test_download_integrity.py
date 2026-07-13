import json
from pathlib import Path
import tempfile
import types
import unittest
import zipfile

from experiments import dl_real_scaled


class DownloadIntegrityTests(unittest.TestCase):
    REVISION = "a" * 40

    @staticmethod
    def _write_archive(path, members):
        with zipfile.ZipFile(path, "w") as archive:
            for name, payload in members:
                archive.writestr(name, payload)

    def _write_manifest(self, root, written_samples):
        payload = {
            "schema_version": 1,
            "dataset_id": dl_real_scaled.DATASET_ID,
            "split": dl_real_scaled.DATASET_SPLIT,
            "requested_revision": self.REVISION,
            "resolved_revision": self.REVISION,
            "requested_samples": written_samples,
            "written_samples": written_samples,
            "archives": {
                "image.zip": {
                    "sha256": dl_real_scaled.sha256_file(root / "image.zip")
                },
                "text.zip": {
                    "sha256": dl_real_scaled.sha256_file(root / "text.zip")
                },
            },
        }
        (root / "download_provenance.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def _valid_pair(self, root):
        self._write_archive(
            root / "image.zip",
            [("images/000001.jpg", b"image-one"), ("images/000002.jpg", b"image-two")],
        )
        self._write_archive(
            root / "text.zip",
            [
                ("celeba-caption/000001.txt", b"caption one"),
                ("celeba-caption/000002.txt", b"caption two"),
            ],
        )
        self._write_manifest(root, written_samples=2)

    def test_valid_pair_is_reusable_and_sha_mutation_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._valid_pair(root)

            status = dl_real_scaled.check_existing_download(2, root, self.REVISION)
            self.assertTrue(status["ready"], status["reasons"])

            self._write_archive(
                root / "text.zip",
                [
                    ("celeba-caption/000001.txt", b"changed"),
                    ("celeba-caption/000002.txt", b"caption two"),
                ],
            )
            status = dl_real_scaled.check_existing_download(2, root, self.REVISION)
            self.assertFalse(status["ready"])
            self.assertIn("text.zip SHA-256", " ".join(status["reasons"]))

    def test_pair_stems_and_manifest_count_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_archive(
                root / "image.zip",
                [("images/000001.jpg", b"one"), ("images/000002.jpg", b"two")],
            )
            self._write_archive(
                root / "text.zip",
                [
                    ("celeba-caption/000001.txt", b"one"),
                    ("celeba-caption/000003.txt", b"three"),
                ],
            )
            self._write_manifest(root, written_samples=3)

            status = dl_real_scaled.check_existing_download(1, root, self.REVISION)

            self.assertFalse(status["ready"])
            reasons = " ".join(status["reasons"])
            self.assertIn("stem sets", reasons)
            self.assertIn("counts", reasons)

    def test_duplicate_archive_stems_are_not_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_archive(
                root / "image.zip",
                [("images/person.jpg", b"one"), ("alternate/person.png", b"two")],
            )
            self._write_archive(
                root / "text.zip",
                [("captions/person.txt", b"one"), ("alternate/person.txt", b"two")],
            )
            self._write_manifest(root, written_samples=2)

            status = dl_real_scaled.check_existing_download(1, root, self.REVISION)

            self.assertFalse(status["ready"])
            self.assertIn("duplicate stems", " ".join(status["reasons"]))

    def test_mutable_refs_resolve_each_time_but_full_sha_skips_lookup(self):
        class FakeApi:
            def __init__(self):
                self.calls = 0

            def dataset_info(self, **_kwargs):
                self.calls += 1
                return types.SimpleNamespace(sha=("b" if self.calls == 1 else "c") * 40)

        api = FakeApi()
        self.assertEqual(dl_real_scaled.resolve_revision("main", api=api), "b" * 40)
        self.assertEqual(dl_real_scaled.resolve_revision("main", api=api), "c" * 40)
        self.assertEqual(api.calls, 2)
        self.assertEqual(
            dl_real_scaled.resolve_revision(self.REVISION, api=api), self.REVISION
        )
        self.assertEqual(api.calls, 2)


if __name__ == "__main__":
    unittest.main()
