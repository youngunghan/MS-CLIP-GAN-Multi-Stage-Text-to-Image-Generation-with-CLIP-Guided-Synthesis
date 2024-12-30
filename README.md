# MS-CLIP-GAN-Multi-Stage-Text-to-Image-Generation-with-CLIP-Guided-Synthesis
This repository presents a novel approach to text-to-image generation that leverages CLIP embeddings in a multi-stage synthesis pipeline, achieving high-quality and semantically consistent image generation from textual descriptions.

## Dataset Preprocessing

### 1. Dataset Structure
The dataset should be organized as follows:
  multimodal_celeba_hq.zip
  ├── images/
  │ ├── 000001.jpg
  │ ├── 000002.jpg
  │ └── ...
  └── celeba-caption/
  ├── 000001.txt
  ├── 000002.txt
  └── ...

### 2. Split Dataset
First, split the dataset into train and test sets:

Give execution permission
chmod +x preprocessing/split_dataset.sh

Run the split script
./preprocessing/split_dataset.sh

This will create two pickle files:
- `celeba_filenames_train.pickle`: Contains training image filenames
- `celeba_filenames_test.pickle`: Contains test image filenames

You can customize the split ratio by modifying the `--train_ratio` parameter in `split_dataset.sh` (default: 0.8).

### 3. Preprocess Dataset
After splitting the dataset, preprocess both train and test sets:

Give execution permission
chmod +x preprocessing/preprocess_train.sh
chmod +x preprocessing/preprocess_test.sh

Process training set
./preprocessing/preprocess_train.sh

Process test set
./preprocessing/preprocess_test.sh

This preprocessing includes:
- Image resizing and cropping to 256x256
- CLIP feature extraction for both images and captions
- Saving processed data in zip format

The preprocessed datasets will be saved as:
- `trainset.zip`: Processed training dataset
- `testset.zip`: Processed test dataset
