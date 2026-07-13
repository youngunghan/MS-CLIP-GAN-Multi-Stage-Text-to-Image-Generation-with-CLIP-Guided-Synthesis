#!/usr/bin/env python3
"""Download a reproducible CelebA-HQ image/caption subset from Hugging Face.

The historical positional sample-count CLI remains valid::

    experiments/dl_real_scaled.py 3000

The default is the dataset commit audited for this project; ``--revision`` can
deliberately select another tag, branch, or commit. In every case the Hub-selected
revision is resolved to a commit SHA before streaming and recorded in
``download_provenance.json``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Optional, Sequence
import zipfile

from PIL import Image


DATASET_ID = "ixw/celebahq-caption-10k"
DATASET_SPLIT = "train"
DEFAULT_REVISION = "50e4e9dc81ca8f974af5a5e088ad82d830a7f5d0"
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "mm-celeba-hq-dataset"
PIL_RESAMPLING = getattr(Image, "Resampling", Image)
COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("sample count must be positive")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count", nargs="?", type=positive_int, default=3000)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=(
            "Hugging Face dataset tag, branch, or commit "
            f"(default: audited commit {DEFAULT_REVISION}; may be overridden)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"raw archive destination (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--resolved-revision",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_revision(requested_revision: Optional[str], api=None) -> str:
    """Resolve a mutable Hub ref on every call; full commit SHAs need no lookup."""
    if requested_revision and COMMIT_SHA_PATTERN.fullmatch(requested_revision):
        return requested_revision.lower()
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    info = api.dataset_info(repo_id=DATASET_ID, revision=requested_revision)
    resolved_revision = info.sha
    if not isinstance(resolved_revision, str) or not COMMIT_SHA_PATTERN.fullmatch(
        resolved_revision
    ):
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {DATASET_ID}")
    return resolved_revision.lower()


def _archive_inventory(path: Path, extensions) -> dict:
    with zipfile.ZipFile(path, mode="r") as archive:
        names = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in extensions
        ]
    stems = [Path(name).stem for name in names]
    duplicates = sorted(stem for stem, occurrences in Counter(stems).items() if occurrences > 1)
    return {
        "count": len(names),
        "stems": set(stems),
        "duplicate_stems": duplicates,
    }


def check_existing_download(count: int, output_dir: Path, resolved_revision: str) -> dict:
    """Validate that an existing raw pair exactly matches its provenance manifest."""
    result = {
        "ready": False,
        "available_samples": 0,
        "resolved_revision": resolved_revision,
        "reasons": [],
    }
    try:
        provenance = json.loads(
            (output_dir / "download_provenance.json").read_text(encoding="utf-8")
        )
        image_path = output_dir / "image.zip"
        text_path = output_dir / "text.zip"
        image = _archive_inventory(image_path, {".jpg", ".jpeg", ".png", ".bmp"})
        text = _archive_inventory(text_path, {".txt"})
        written_samples = int(provenance["written_samples"])
        stored_image_sha = provenance["archives"]["image.zip"]["sha256"]
        stored_text_sha = provenance["archives"]["text.zip"]["sha256"]
        actual_image_sha = sha256_file(image_path)
        actual_text_sha = sha256_file(text_path)
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        result["reasons"].append(f"unreadable or incomplete download: {exc}")
        return result

    result["available_samples"] = min(
        image["count"], text["count"], written_samples
    )
    checks = [
        (
            provenance.get("schema_version") == 1,
            "provenance schema version does not match",
        ),
        (
            provenance.get("dataset_id") == DATASET_ID,
            "dataset id does not match",
        ),
        (
            provenance.get("split") == DATASET_SPLIT,
            "dataset split does not match",
        ),
        (
            provenance.get("resolved_revision") == resolved_revision,
            "resolved revision does not match",
        ),
        (
            stored_image_sha == actual_image_sha,
            "image.zip SHA-256 does not match provenance",
        ),
        (
            stored_text_sha == actual_text_sha,
            "text.zip SHA-256 does not match provenance",
        ),
        (
            not image["duplicate_stems"],
            f"image.zip has duplicate stems: {image['duplicate_stems']}",
        ),
        (
            not text["duplicate_stems"],
            f"text.zip has duplicate stems: {text['duplicate_stems']}",
        ),
        (
            image["stems"] == text["stems"],
            "image/text stem sets do not match",
        ),
        (
            image["count"] == text["count"] == written_samples,
            "archive and manifest counts do not match exactly",
        ),
        (
            written_samples >= count,
            f"only {written_samples} samples are available; {count} requested",
        ),
    ]
    result["reasons"].extend(message for passed, message in checks if not passed)
    result["ready"] = not result["reasons"]
    return result


def _image_from_row(value) -> Image.Image:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as source:
                return source.convert("RGB")
        if value.get("path"):
            with Image.open(value["path"]) as source:
                return source.convert("RGB")
        raise ValueError("image mapping contains neither bytes nor path")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    raise TypeError(f"unsupported image value: {type(value).__name__}")


def _atomic_json_dump(payload, path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def download(
    count: int,
    output_dir: Path,
    requested_revision: Optional[str],
    resolved_revision: Optional[str] = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Resolve the selected branch/tag to immutable content before loading and
    # record the Hub-returned SHA. The default is an audited commit, while callers
    # can deliberately select another revision with --revision.
    if resolved_revision is None:
        resolved_revision = resolve_revision(requested_revision)
    elif not COMMIT_SHA_PATTERN.fullmatch(resolved_revision):
        raise ValueError("resolved_revision must be a full 40-character commit SHA")
    else:
        resolved_revision = resolved_revision.lower()

    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        split=DATASET_SPLIT,
        streaming=True,
        revision=resolved_revision,
    )

    image_final = output_dir / "image.zip"
    text_final = output_dir / "text.zip"
    image_tmp = output_dir / f".image.{os.getpid()}.tmp"
    text_tmp = output_dir / f".text.{os.getpid()}.tmp"
    written = 0
    fallback_captions = 0
    try:
        with zipfile.ZipFile(image_tmp, "w", zipfile.ZIP_STORED) as image_zip, zipfile.ZipFile(
            text_tmp, "w", zipfile.ZIP_DEFLATED
        ) as text_zip:
            for row in dataset:
                if written >= count:
                    break
                image = _image_from_row(row["image"])
                if image.size != (256, 256):
                    image = image.resize((256, 256), PIL_RESAMPLING.BICUBIC)
                image_bytes = io.BytesIO()
                image.save(image_bytes, "JPEG", quality=95)

                caption = (row.get("text") or "").strip()
                if not caption:
                    caption = "a photography of a person"
                    fallback_captions += 1

                written += 1
                image_zip.writestr(f"images/{written:06d}.jpg", image_bytes.getvalue())
                text_zip.writestr(f"celeba-caption/{written:06d}.txt", caption)
                if written % 500 == 0:
                    print(f"  ... {written}/{count}", flush=True)

        if written != count:
            raise RuntimeError(
                f"dataset ended after {written} samples; requested {count}; existing archives were not replaced"
            )

        # NOTE: these SHA-256 digests are an *artifact*-integrity check over the
        # locally re-encoded image.zip/text.zip produced by this run (JPEG
        # re-encoding, quality=95, and zip packing are not guaranteed
        # byte-stable across library versions/platforms) -- they are not a
        # *source*-provenance hash of the upstream Hugging Face dataset. Source
        # provenance is instead tracked via `resolved_revision`, the immutable
        # commit SHA the Hub resolved the requested tag/branch/commit to.
        image_sha256 = sha256_file(image_tmp)
        text_sha256 = sha256_file(text_tmp)
        os.replace(image_tmp, image_final)
        os.replace(text_tmp, text_final)
    finally:
        for temporary in (image_tmp, text_tmp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    provenance = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "split": DATASET_SPLIT,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "requested_samples": count,
        "written_samples": written,
        "fallback_caption_count": fallback_captions,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archives": {
            "image.zip": {"sha256": image_sha256},
            "text.zip": {"sha256": text_sha256},
        },
        "package_versions": {
            "datasets": importlib.metadata.version("datasets"),
            "huggingface_hub": importlib.metadata.version("huggingface_hub"),
            "pillow": importlib.metadata.version("Pillow"),
        },
    }
    _atomic_json_dump(provenance, output_dir / "download_provenance.json")
    return provenance


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.resolved_revision is not None:
        if not COMMIT_SHA_PATTERN.fullmatch(args.resolved_revision):
            raise ValueError("--resolved-revision must be a full 40-character commit SHA")
        resolved_revision = args.resolved_revision.lower()
    else:
        resolved_revision = resolve_revision(args.revision)
    if args.check_existing:
        status = check_existing_download(args.count, args.output_dir, resolved_revision)
        print(
            f"{int(status['ready'])}\t{status['available_samples']}\t"
            f"{status['resolved_revision']}"
        )
        return 0

    # Streaming re-downloads the whole dataset from scratch on every call (no
    # byte-range resume), so guard against redundant re-downloads: if a prior
    # run already left a verified download at this resolved revision covering
    # at least the requested --count (check_existing_download uses >=, not
    # ==), skip straight to reuse instead of re-streaming it.
    existing = check_existing_download(args.count, args.output_dir, resolved_revision)
    if existing["ready"]:
        print(
            f"skip download: {existing['available_samples']} verified samples already "
            f"present at revision {resolved_revision} -> {args.output_dir}",
            flush=True,
        )
        return 0

    provenance = download(
        args.count,
        args.output_dir,
        args.revision,
        resolved_revision=resolved_revision,
    )
    print(
        f"downloaded {provenance['written_samples']} REAL samples at "
        f"revision {provenance['resolved_revision']} -> {args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
