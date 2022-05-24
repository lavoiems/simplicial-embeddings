#!/bin/bash

module load python/3.8

source ~/env/bin/activate

python ../../../main_pretrain.py \
    --dataset cifar100 \
    --backbone resnet50 \
    --data_dir ./datasets \
    --checkpoint_dir $SCRATCH/sem_nips \
    --max_epochs 1000 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --classifier_lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name sdmoco \
    --project PROJECT \
    --entity ENTITY \
    --wandb \
    --save_checkpoint \
    --method sdmoco \
    --proj_hidden_dim 2048 \
    --queue_size 32768 \
    --temperature 0.2 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 0.999 \
    --momentum_classifier \
    --voc_size 13 \
    --message_size 5000 \
    --tau_online $1 \
    --tau_target $1
