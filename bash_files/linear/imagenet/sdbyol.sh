#!/bin/bash

ROOT_PATH=$1

../../../prepare_data.sh TEST

module load python/3.8
source ~/env/bin/activate

python3 ../../../main_linear.py \
    --dataset imagenet \
    --backbone resnet50 \
    --checkpoint_dir=$ROOT_PATH \
    --data_dir $SLURM_TMPDIR \
    --train_dir $SLURM_TMPDIR/data/train \
    --val_dir   $SLURM_TMPDIR/data/val \
    --max_epochs 100 \
    --gpus 0,1 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --dali \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 512 \
    --num_workers 10 \
    --pretrained_feature_extractor $2 \
    --name byol-resnet50-imagenet-linear-eval \
    --entity il_group \
    --project VIL \
    --save_checkpoint \
    --taus 2 3 \
    --lrs 0.1 0.05 \
    --wd1 0  \
    --wd2 1e-6 1e-5 \
    --wandb

