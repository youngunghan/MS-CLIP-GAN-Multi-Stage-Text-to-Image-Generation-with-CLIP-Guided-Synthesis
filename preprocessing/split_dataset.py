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
            
            # Filter image files and get their base names
            image_files = [
                Path(f).stem for f in all_files 
                if f.startswith('images/') and
                any(f.lower().endswith(ext) for ext in ['.jpg', '.png', '.jpeg'])
            ]
            print("DEBUG: First 10 filtered files:", image_files[:10])
    else:
        # Original directory-based logic
        image_dir = Path(os.path.join(source_path, 'images'))
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            image_files.extend([f.stem for f in image_dir.rglob(ext)])
    
    return sorted(image_files)

def split_dataset(source_path, train_ratio=0.8, seed=42):
    """Split dataset into train and test sets"""
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all image files
    all_files = get_all_image_files(source_path)
    print(f"Total number of images found: {len(all_files)}")
    
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
    
    # Save train files
    train_pickle_path = os.path.join(project_root, 'data', 'celeba_filenames_train.pickle')
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
    
    args = parser.parse_args()
    
    split_dataset(args.source_path, args.train_ratio, args.seed) 