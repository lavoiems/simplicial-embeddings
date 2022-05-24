#!/bin/bash

module load python/3.8
source ~/class_env/bin/activate

cp -r datasets $SLURM_TMPDIR

python ../../../main_linear.py \
    --dataset cifar100 \
    --backbone resnet50 \
    --data_dir $SLURM_TMPDIR/datasets \
    --checkpoint_dir $2 \
    --pretrained_feature_extractor $1 \
    --max_epochs 100 \
    --gpus 0 \
    --accelerator gpu \
    --save_checkpoint \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --method linear \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 5 \
    --crop_size 32 \
    --name class_sdmoco \
    --lrs 0.05 \
    --wd1 1e-6 \
    --wd2 0 \
    --class_base False \
    ${@:3}

    #--wd1 0 1e-6 1e-5 1e-4 \
