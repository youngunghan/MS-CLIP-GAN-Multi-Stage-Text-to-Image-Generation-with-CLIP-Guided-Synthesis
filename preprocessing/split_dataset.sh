#!/bin/bash

python preprocessing/split_dataset.py \
    --source_path /home/yuhan/test/Text-to-Image-generation-main/data/mm-celeba-hq-dataset \
    --train_ratio 0.85 \
    --seed 42
