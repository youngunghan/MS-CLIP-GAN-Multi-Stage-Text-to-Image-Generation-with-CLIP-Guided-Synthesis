# MS-CLIP-GAN: Multi-Stage Text-to-Image Generation with CLIP-Guided Synthesis

This repository implements an experimental multi-stage text-to-image GAN that combines
CLIP text embeddings with 64→128→256 synthesis. It is a research prototype assembled
from ideas used by StackGAN++, LAFITE, AttnGAN, and related work; it does not claim a new
state of the art. The checked-in experiment record is limited to a single-seed,
small-subset study, and its samples remain visibly blurry/distorted. Read
[the correctness and limitations note](docs/explanation/correctness-and-fixes.md) before
quoting results.

---

## Documentation

Full developer docs live in [docs/README.md](docs/README.md) (human index) and [docs/llms.txt](docs/llms.txt) (LLM index), organized by Diátaxis (tutorials / how-to / reference / explanation):

- Quickstart: [docs/tutorials/quickstart.md](docs/tutorials/quickstart.md)
- Dataset preparation: [docs/how-to/prepare-dataset.md](docs/how-to/prepare-dataset.md)
- Architecture (with figures): [docs/explanation/architecture.md](docs/explanation/architecture.md)
- Correctness & known limitations: [docs/explanation/correctness-and-fixes.md](docs/explanation/correctness-and-fixes.md)

## Dataset

This project uses **MM-CelebA-HQ**. Download it from the original distribution (MM-CelebA-HQ-Dataset) and place it as a directory containing `image.zip` and `text.zip` (see the structure below). The preprocessed output zip schema is documented in [docs/reference/dataset-format.md](docs/reference/dataset-format.md).

---

## Dataset Preprocessing

### 0. Environment Setup
Before preprocessing the dataset, ensure you have the required environment set up:

1. Create Conda Environment:
   ```bash
   conda env create -f environment.yml
   ```

   To update an existing `msclipgan` environment, run
   `conda env update -f environment.yml --prune`. The audited core combination is
   Python 3.8.20, PyTorch 2.4.0, TorchVision 0.19.0, and CUDA runtime 12.4.

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
data/mm-celeba-hq-dataset/
├── image.zip          # images/000001.jpg, images/000002.jpg, ...
└── text.zip           # celeba-caption/000001.txt, celeba-caption/000002.txt, ...
```

> The preprocessing scripts take `--source ./data/mm-celeba-hq-dataset` (a directory containing `image.zip` and `text.zip`), not a single archive file. See [docs/how-to/prepare-dataset.md](docs/how-to/prepare-dataset.md).

### 2. Split Dataset
To split the dataset into training and testing sets:

1. **Run the split script:**
   ```bash
   bash preprocessing/split_dataset.sh
   ```

This will generate two pickle files:
- `celeba_filenames_train.pickle`: Contains filenames for the training set.
- `celeba_filenames_test.pickle`: Contains filenames for the test set.

You can customize the split ratio by modifying the `--train_ratio` parameter in `split_dataset.sh` (default: `0.85`).

### 3. Preprocess Dataset
After splitting the dataset, preprocess both training and testing sets:

1. **Process the training set:**
   ```bash
   bash preprocessing/preprocess_train.sh
   ```

2. **Process the testing set:**
   ```bash
   bash preprocessing/preprocess_test.sh
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

1. **Run the training script:**
   ```bash
   # BS=4 measured about 5.8–6.2 GB on an RTX 4060 Ti 8 GB.
   bash train.sh
   ```

This starts a fresh run with the audited linear-conditioning and image-only-alignment defaults. The linear choice addresses saturation measured in this repository's legacy checkpoint; metadata-less historical checkpoints still use their original ReLU behavior. Training saves at the configured frequency and always saves the final epoch. Configure `GPUS` for multi-GPU use. For exact resume versus intentional schedule extension, follow [docs/how-to/train-eval-infer.md](docs/how-to/train-eval-infer.md); do not execute the two commented resume flags as a separate shell command.

Metadata-less historical checkpoints are loaded with their legacy conditioning/alignment semantics. They remain usable for evaluation and inference, but evaluating the audited fresh-run defaults requires retraining.

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

1. **Run the evaluation script:**
   ```bash
   # bash eval.sh <CKPT_DIR> [EPOCH]
   # CKPT_DIR (required) is the directory containing epoch_<EPOCH>_Gen.pt
   bash eval.sh ./checkpoints/<run_name>/ckpt 149
   ```

This will start the evaluation process using the specified parameters. The `--prompt` text inside `eval.sh` is a shared-parser placeholder and is not used by dataset evaluation.

---

## Inference

#### 0. Environment Setup
Ensure you have the required environment set up as described in the "Environment Setup" section above.

#### 1. Prepare Input Text
Prepare the text descriptions you want to use for generating images.

#### 2. Run the Inference Script
Create and execute an inference script to generate images from text:

1. **Run the inference script with its built-in prompt:**
   ```bash
   # bash infer.sh <CKPT_DIR> [EPOCH]
   # CKPT_DIR (required) is the directory containing epoch_<EPOCH>_Gen.pt
   bash infer.sh ./checkpoints/<run_name>/ckpt 149
   ```

This will generate images based on the provided text descriptions using the specified checkpoint.
Only the generator checkpoint is required for inference; discriminator checkpoints are not needed.

`infer.sh` accepts only `<CKPT_DIR> [EPOCH]`; it does not accept a prompt as a third positional argument. To choose a prompt, call the Python entrypoint:

```bash
PYTHONPATH=. python scripts/infer.py \
  --checkpoint_path ./checkpoints/<run_name>/ckpt \
  --load_epoch 149 \
  --eval_data_path None \
  --prompt "a portrait of a person with blond hair"
```

---

## References

- Dataset: The dataset is based on MM-CelebA-HQ-Dataset, which provides multi-modal data including images, captions, semantic masks, and sketches.
- Dataset Preprocessing Code: The preprocessing scripts are adapted from StyleGAN2-ADA-PyTorch.

---
