export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/eval.py \
    --prompt The woman is young and has blond hair, and arched eyebrows. \
    --load_epoch 99 \
    --eval_data_path /home/yuhan/test/Text-to-Image-generation-main/data/testset.zip \
    --checkpoint_path /home/yuhan/test/Text-to-Image-generation-main/checkpoints/msclipgan_bs128_lr1e-4_epoch100-2025_01_08_07_47_55/ckpt
