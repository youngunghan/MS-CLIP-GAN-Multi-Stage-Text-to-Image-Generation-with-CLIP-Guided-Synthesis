#!/bin/bash

# Test dataset preprocessing
python preprocessing/preprocess_dataset.py \
    --source /home/yuhan/test/Text-to-Image-generation-main/data/mm-celeba-hq-dataset \
    --src_data_list /home/yuhan/test/Text-to-Image-generation-main/data/celeba_filenames_test.pickle \
    --dest /home/yuhan/test/Text-to-Image-generation-main/data/testset.zip \
    --transform=center-crop \
    --width=256 \
    --height=256 \
    --emb_dim=512