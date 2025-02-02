#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Training hyperparameters
BS=64
LR=1e-4
EPOCH=150
SAVE_FREQ=1  # 5 에폭마다 저장

# GPU settings
GPUS="1,2"  # Using GPU 0 and 3
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

# Calculate effective batch size
EFFECTIVE_BS=$((BS * NUM_GPUS))

# Generate dynamic experiment name
DATE=$(date +%Y_%m_%d_%H_%M_%S)
EXP_NAME="msclipgan_bs${EFFECTIVE_BS}_lr${LR}_epoch${EPOCH}_${DATE}"

# Run training
CUDA_VISIBLE_DEVICES=$GPUS python scripts/train.py \
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
    --num_workers $((NUM_GPUS * 4)) \
    --resume_epoch 99 \
    --resume_checkpoint_path /home/yuhan/test/Text-to-Image-generation-main/checkpoints/msclipgan_bs128_lr1e-4_epoch100_2025_01_30_22_20_21-2025_01_30_22_20_24/ckpt/msclipgan_bs128_lr1e-4_epoch100_2025_01_30_22_20_21-2025_01_30_22_20_24
#--num_workers $((NUM_GPUS * 4)) \# Adjust number of workers based on GPU count
# parser.add_argument('--name', type=str, default='experiment_name', 
#                             help='name of the experiment. It decides where to store samples and models')
#         parser.add_argument('--gpu_ids', type=str, default='0', 
#                             help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
#         parser.add_argument('--num_workers', type=int, default=4)

#         parser.add_argument('--data_path', default='./data/sample_train.zip', type=Path, help="path of directory containing training dataset")  
#         parser.add_argument('--resume_checkpoint_path', default=None)
#         parser.add_argument('--resume_epoch', type=int, default=-1)
#         parser.add_argument('--report_interval', type=int, default=100, help='Report interval')
#         parser.add_argument('--checkpoint_path', type=Path, default='./checkpoints', help='Checkpoint path')
#         parser.add_argument('--result_path', type=Path, default='./output', help='Generated image path')     
        
#         parser.add_argument('--noise_dim', type=int, default=100, help= 'Input noise dimension to Generator')
#         parser.add_argument('--condition_dim', type=int, default=128, help= 'Noise projection dimension')
#         parser.add_argument('--clip_embedding_dim', type=int, default=512, help= 'Dimension of c_txt from CLIP ViT-B/32')
        
#         parser.add_argument('--g_in_chans', type=int, default=1024,
#                             help='Number of input channels for generator (Ng)')                            
#         parser.add_argument('--g_out_chans', type=int, default=3,
#                             help='Number of output channels for generator')                            
#         parser.add_argument('--d_in_chans', type=int, default=64,
#                             help='Number of input channels for discriminator (Nd)')                            
#         parser.add_argument('--d_out_chans', type=int, default=1,
#                             help='Number of output channels for discriminator')
#         parser.add_argument('--num_stage', type=int, default=3)      
        
#         parser.add_argument('--clip_model', type=str, choices=(['ViT-B/32', 'ViT-L/14', 'ViT-B/16']), default='ViT-B/32')
# parser.add_argument('--batch_size', type=int, default=1, help='Batch size') #64
#         parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
#         parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')   
#         parser.add_argument('--use_uncond_loss', action="store_true")
#         parser.add_argument('--use_contrastive_loss', action="store_true")
#         parser.add_argument('--use_mixed_loss', action="store_true")
#         parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')