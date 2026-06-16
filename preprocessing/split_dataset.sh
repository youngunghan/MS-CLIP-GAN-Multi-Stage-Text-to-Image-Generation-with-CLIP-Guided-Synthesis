#!/bin/bash

python preprocessing/split_dataset.py \
    --source_path ./data/mm-celeba-hq-dataset \
    --train_ratio 0.85 \
    --seed 42
