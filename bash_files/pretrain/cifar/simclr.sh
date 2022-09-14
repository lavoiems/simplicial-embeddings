#!/bin/bash

module load python/3.8

source env/bin/activate

echo "eta_lr 0.02"

python ../../../main_pretrain.py \
    --dataset cifar100 \
    --checkpoint_dir /network/scratch/l/lavoiems/baselines \
    --backbone resnet18 \
    --data_dir ./datasets \
    --max_epochs 1000 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 4 \
    --crop_size 32 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name baselines-simclr \
    --project iclr \
    --entity il_group \
    --wandb \
    --save_checkpoint \
    --method simclr \
    --temperature 0.2 \
    --proj_hidden_dim 2048 \
    --proj_output_dim 256
