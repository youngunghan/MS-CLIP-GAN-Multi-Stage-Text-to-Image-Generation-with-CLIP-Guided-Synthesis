import os
import pickle
import random
import zipfile
from pathlib import Path

def get_all_image_files(source_path):
    """Get all image files from the source directory or zip file"""
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
            image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            image_files = [
                Path(f).stem for f in all_files
                if not f.endswith('/') and f.lower().endswith(image_exts)
            ]
            print("DEBUG: First 10 filtered files:", image_files[:10])
    else:
        # Original directory-based logic
        image_dir = Path(os.path.join(source_path, 'images'))
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            image_files.extend([f.stem for f in image_dir.rglob(ext)])

    return sorted(image_files)

def split_dataset(source_path, train_ratio=0.8, seed=42, max_images=None):
    """Split dataset into train and test sets"""
    # Set random seed for reproducibility
    random.seed(seed)

    # Get all image files
    all_files = get_all_image_files(source_path)
    print(f"Total number of images found: {len(all_files)}")

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