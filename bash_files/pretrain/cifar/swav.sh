#!/bin/bash

module load python/3.8

source ~/env_1.12/bin/activate

python3 ../../../main_pretrain.py \
    --dataset cifar100 \
    --backbone resnet50 \
    --data_dir ./datasets \
    --max_epochs 1000 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.001 \
    --sk_epsilon 0.03 \
    --scheduler warmup_cosine \
    --lr 0.1 \
    --min_lr 0.0006 \
    --classifier_lr 0.1 \
    --weight_decay 1e-6 \
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
    --name swav-cifar100 \
    --wandb \
    --save_checkpoint \
    --project VIL \
    --entity il_group \
    --method swav \
    --proj_hidden_dim 2048 \
    --queue_size 3840 \
    --proj_output_dim 128 \
    --num_prototypes 3000 \
    --epoch_queue_starts 50 \
    --freeze_prototypes_epochs 2

