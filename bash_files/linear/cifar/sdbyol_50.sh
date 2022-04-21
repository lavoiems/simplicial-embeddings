#!/bin/bash

module load python/3.8
source ~/env/bin/activate

cp -r datasets $SLURM_TMPDIR

python ../../../main_linear.py \
    --dataset cifar100 \
    --backbone resnet50 \
    --data_dir $SLURM_TMPDIR/datasets \
    --max_epochs 100 \
    --gpus 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 5 \
    --crop_size 32 \
    --name class_sdbyol \
    --pretrained_feature_extractor $1 \
    --taus 0.0001 0.001 0.01 0.1 1 10 \
    --eval_taus 0.0001 0.001 0.01 0.1 1 10 \
    --lrs 0.1 0.01 \
    --wd1 0 1e-6 1e-5 \
    --wd2 0 1e-6 1e-5 \
    --class_base True \
    --name sdbyol \
    ${@:2}

