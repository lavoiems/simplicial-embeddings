#!/bin/bash

module load python/3.8

source /home/mila/l/lavoiems/env/bin/activate

python ../../../main_pretrain.py \
    --dataset cifar100 \
    --backbone resnet50 \
    --data_dir ./datasets \
    --checkpoint_dir $SCRATCH/cifar_rn50_scale \
    --max_epochs 1000 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.5 \
    --classifier_lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --num_workers 6 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --solarization_prob 0.0 0.2 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name sdbyol-z:4096 \
    --project syn_ssl \
    --entity lavoiems \
    --wandb \
    --save_checkpoint \
    --method sdbyol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier \
    --voc_size $2 \
    --message_size $1 \
    --tau_online $3 \
    --tau_target $3 \

