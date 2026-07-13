#!/usr/bin/env python3

import os
import pickle
import random
import zipfile
from pathlib import Path

import PIL.Image


def _file_ext(name) -> str:
    return str(name).split('.')[-1]


def is_image_ext(fname) -> bool:
    """Match preprocess_dataset.py's is_image_ext(): any extension PIL can
    decode. Both scripts must agree on this policy -- a stem that one accepts
    and the other silently ignores turns into a spurious duplicate-stem error
    or a split/preprocess mismatch downstream."""
    ext = _file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION  # type: ignore


def _unique_stems(image_paths, source_label):
    """Return sorted stems, rejecting identities that map to multiple images."""
    by_stem = {}
    for image_path in image_paths:
        by_stem.setdefault(Path(image_path).stem, []).append(str(image_path))
    collisions = {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}
    if collisions:
        details = "; ".join(
            f"{stem}: {', '.join(paths)}" for stem, paths in sorted(collisions.items())
        )
        raise ValueError(f"duplicate image stems in {source_label}: {details}")
    return sorted(by_stem)


def get_all_image_files(source_path):
    """Get all image files from the source directory or zip file"""
    PIL.Image.init()  # type: ignore  # populate PIL.Image.EXTENSION before filtering
    image_files = []

    # Check if source is a directory containing image.zip
    image_zip = os.path.join(source_path, 'image.zip')
    if os.path.isfile(image_zip):
        with zipfile.ZipFile(image_zip, 'r') as z:
            # Get all files from zip
            all_files = z.namelist()
            print("DEBUG: First 10 files in zip:", all_files[:10])

            # Filter image files by extension (do NOT assume an 'images/' prefix; the
            # archive may store images at top level, under image/ or images/, etc.)
            image_paths = [
                f for f in all_files
                if not f.endswith('/') and is_image_ext(f)
            ]
            print("DEBUG: First 10 filtered files:", image_paths[:10])
            image_files = _unique_stems(image_paths, image_zip)
    else:
        # Original directory-based logic
        image_dir = Path(os.path.join(source_path, 'images'))
        image_paths = [
            path for path in image_dir.rglob('*')
            if path.is_file() and is_image_ext(path)
        ]
        image_files = _unique_stems(image_paths, str(image_dir))

    return image_files

def split_dataset(source_path, train_ratio=0.8, seed=42, max_images=None):
    """Split dataset into train and test sets"""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1 (exclusive); got {train_ratio}")
    if max_images is not None and max_images <= 0:
        raise ValueError(f"max_images must be positive when provided; got {max_images}")

    # Set random seed for reproducibility
    random.seed(seed)

    # Get all image files
    all_files = get_all_image_files(source_path)
    print(f"Total number of images found: {len(all_files)}")
    if not all_files:
        raise ValueError(f"no input images found under {source_path}")

    # Cap BEFORE shuffling. The list is sorted and the HF downloader writes the first
    # N samples in order, so sorted[:N] equals what a fresh N-image download would
    # contain. Without this cap, an image.zip left over from a larger prep silently
    # inflates the split (e.g. a "3000-image subset" becomes all 10000 images).
    if max_images is not None and max_images > 0 and len(all_files) > max_images:
        all_files = all_files[:max_images]
        print(f"Capped to the first {max_images} images (--max_images)")

    # Shuffle the files
    random.shuffle(all_files)

    # Split into train and test
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    if not train_files or not test_files:
        raise ValueError(
            f"split would be empty: {len(train_files)} train / {len(test_files)} test; "
            "provide at least two images or adjust --train_ratio"
        )
    if len(train_files) + len(test_files) != len(all_files):
        raise RuntimeError("split count mismatch")

    print(f"Number of training images: {len(train_files)}")
    print(f"Number of test images: {len(test_files)}")

    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Ensure the data/ output directory exists
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Save train files
    train_pickle_path = os.path.join(data_dir, 'celeba_filenames_train.pickle')
    with open(train_pickle_path, 'wb') as f:
        pickle.dump(train_files, f)
    print(f"Saved train files to {train_pickle_path}")

    # Save test files
    test_pickle_path = os.path.join(project_root, 'data', 'celeba_filenames_test.pickle')
    with open(test_pickle_path, 'wb') as f:
        pickle.dump(test_files, f)
    print(f"Saved test files to {test_pickle_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Split dataset into train and test sets')
    parser.add_argument('--source_path', type=str, required=True,
                        help='Path to the dataset directory or zip file')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Use only the first N images (sorted order) before splitting')

    args = parser.parse_args()

    split_dataset(args.source_path, args.train_ratio, args.seed, args.max_images)
