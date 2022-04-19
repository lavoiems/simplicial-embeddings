#!/bin/bash

ROOT_PATH=$1

../../../prepare_data.sh TEST

module load python/3.8
source ~/env_pt11/bin/activate

python3 ../../../main_pretrain.py \
    --dataset imagenet \
    --backbone resnet50 \
    --checkpoint_dir=$ROOT_PATH \
    --data_dir $SLURM_TMPDIR/data \
    --train_dir $SLURM_TMPDIR/data/train/ \
    --val_dir   $SLURM_TMPDIR/data/val/ \
    --max_epochs 800 \
    --gpus 0,1,2,3 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.45 \
    --accumulate_grad_batches 8 \
    --classifier_lr 0.2 \
    --weight_decay 1e-6 \
    --batch_size 128 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --name "sdbyol-L:4500-ep:800" \
    --entity il_group \
    --project sem_neurips \
    --wandb \
    --offline \
    --save_checkpoint \
    --method sdbyol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --auto_resume \
    --voc_size 21 \
    --message_size 4500 \
    --tau_online 0.872 \
    --tau_target 0.159 \
    --momentum_classifier
