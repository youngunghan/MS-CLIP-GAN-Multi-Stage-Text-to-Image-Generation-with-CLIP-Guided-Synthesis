# MS-CLIP-GAN: Multi-Stage Text-to-Image Generation with CLIP-Guided Synthesis

This repository introduces a novel approach to text-to-image generation that utilizes CLIP embeddings in a multi-stage synthesis pipeline. The method achieves high-quality and semantically consistent image generation from textual descriptions.

---

## Dataset Download and Other Details
Instructions for downloading the dataset and additional details are provided at the bottom of this README file.

---

## Dataset Preprocessing

### 0. Environment Setup
Before preprocessing the dataset, ensure you have the required environment set up:

1. Create Conda Environment:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the Environment:
   ```bash
   conda activate msclipgan
   ```

3. Install Additional Dependencies (Linux Only):
   If you encounter the error `ImportError: libGL.so.1: cannot open shared object file`, install the following:
   ```bash
   sudo apt-get install libgl1-mesa-glx
   ```

### 1. Dataset Structure
The dataset should be organized in the following structure:

```
data/MM-Celeba-HQ-Dataset.zip
├── image.zip/
├── text.zip/
├── image/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── text/
    ├── 000001.txt
    ├── 000002.txt
    └── ...
```

### 2. Split Dataset
To split the dataset into training and testing sets:

1. **Give execution permission to the script:**
   ```bash
   chmod +x preprocessing/split_dataset.sh
   ```

2. **Run the split script:**
   ```bash
   ./preprocessing/split_dataset.sh
   ```

This will generate two pickle files:
- `celeba_filenames_train.pickle`: Contains filenames for the training set.
- `celeba_filenames_test.pickle`: Contains filenames for the test set.

You can customize the split ratio by modifying the `--train_ratio` parameter in `split_dataset.sh` (default: `0.8`).

### 3. Preprocess Dataset
After splitting the dataset, preprocess both training and testing sets:

1. **Give execution permission to the preprocessing scripts:**
   ```bash
   chmod +x preprocessing/preprocess_train.sh
   chmod +x preprocessing/preprocess_test.sh
   ```

2. **Process the training set:**
   ```bash
   ./preprocessing/preprocess_train.sh
   ```

3. **Process the testing set:**
   ```bash
   ./preprocessing/preprocess_test.sh
   ```

#### Preprocessing Steps:
- Resizing and cropping images to `256x256`.
- Extracting CLIP features for both images and captions.
- Saving processed data in zip format.

#### Output:
The preprocessed datasets will be saved as:
- `trainset.zip`: Processed training dataset.
- `testset.zip`: Processed test dataset.

---

## Train

### 0. Environment Setup
Ensure you have the required environment set up as described in the "Environment Setup" section above.

### 1. Prepare the Dataset
Make sure you have preprocessed the dataset as described in the "Dataset Preprocessing" section. You should have `trainset.zip` ready for training.

### 2. Run the Training Script
Execute the training script to start the training process:

1. **Give execution permission to the training script:**
   ```bash
   chmod +x train.sh
   ```

2. **Run the training script:**
   ```bash
   ./train.sh
   ```

This will start the training process using the specified parameters. The script will automatically handle multi-GPU settings and save checkpoints at the specified frequency.

### 3. Monitor Training
You can monitor the training process using TensorBoard. The logs are saved in the `runs` directory.

```bash
tensorboard --logdir=runs
```

---
## Evaluation

#### 0. Environment Setup
Ensure you have the required environment set up as described in the "Environment Setup" section above.

#### 1. Prepare the Dataset
Make sure you have preprocessed the dataset as described in the "Dataset Preprocessing" section. You should have `testset.zip` ready for evaluation.

#### 2. Run the Evaluation Script
Execute an evaluation script to start the evaluation process:

1. **Give execution permission to the evaluation script:**
   ```bash
   chmod +x eval.sh
   ```

2. **Run the evaluation script:**
   ```bash
   ./eval.sh
   ```

This will start the evaluation process using the specified parameters.

---

## Inference

#### 0. Environment Setup
Ensure you have the required environment set up as described in the "Environment Setup" section above.

#### 1. Prepare Input Text
Prepare the text descriptions you want to use for generating images.

#### 2. Run the Inference Script
Create and execute an inference script to generate images from text:

1. **Give execution permission to the inference script:**
   ```bash
   chmod +x infer.sh
   ```

2. **Run the inference script:**
   ```bash
   ./infer.sh
   ```

This will generate images based on the provided text descriptions using the specified checkpoint.

---

## References

- Dataset: The dataset is based on MM-CelebA-HQ-Dataset, which provides multi-modal data including images, captions, semantic masks, and sketches.
- Dataset Preprocessing Code: The preprocessing scripts are adapted from StyleGAN2-ADA-PyTorch.

---
