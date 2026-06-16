#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Training hyperparameters
BS=64
LR=1e-4
EPOCH=150
SAVE_FREQ=5            # save checkpoints/samples every N epochs

# GPU settings: comma-separated CUDA device indices passed straight to --gpu_ids.
# (We do NOT set CUDA_VISIBLE_DEVICES here, so --gpu_ids are the real device indices.)
GPUS="0"
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

# Experiment name (base_options.py appends a timestamp automatically)
EXP_NAME="msclipgan_bs${BS}_lr${LR}_epoch${EPOCH}"

# Train from scratch. --gpu_ids selects the CUDA device(s) directly.
python scripts/train.py \
    --name $EXP_NAME \
    --batch_size $BS \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --save_freq $SAVE_FREQ \
    --data_path ./data/trainset.zip \
    --use_uncond_loss \
    --use_contrastive_loss \
    --use_mixed_loss \
    --gpu_ids $GPUS \
    --num_workers $((NUM_GPUS * 4))

# To resume training, append:
#   --resume_checkpoint_path ./checkpoints/<run_name>/ckpt \
#   --resume_epoch <epoch>
