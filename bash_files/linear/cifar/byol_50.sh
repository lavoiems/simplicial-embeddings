#!/bin/bash

module load python/3.8
source ~/env/bin/activate

cp -r datasets $SLURM_TMPDIR

python ../../../main_linear.py \
    --linear_base True \
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
    --method linear \
    --optimizer sgd \
    --scheduler step \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 5 \
    --crop_size 32 \
    --name class_sdbyol \
    --class_base True \
    --name byol \
    ${@:3}

