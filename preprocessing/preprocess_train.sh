#!/bin/bash

# Train dataset preprocessing
python preprocessing/preprocess_dataset.py \
    --source ./data/mm-celeba-hq-dataset \
    --src_data_list ./data/celeba_filenames_train.pickle \
    --dest ./data/trainset.zip \
    --transform=center-crop \
    --width=256 \
    --height=256 \
    --emb_dim=512
