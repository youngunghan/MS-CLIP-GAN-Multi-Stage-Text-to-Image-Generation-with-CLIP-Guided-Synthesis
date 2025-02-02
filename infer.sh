export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/infer.py \
    --prompt The woman is young and has blond hair, and arched eyebrows. \
    --load_epoch 99 \
    --eval_data_path None \
    --checkpoint_path /home/yuhan/test/Text-to-Image-generation-main/checkpoints/msclipgan_bs128_lr1e-4_epoch100-2025_01_08_07_47_55/ckpt