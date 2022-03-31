#!/bin/bash

ROOT_PATH=$1

#../../../prepare_data.sh VAL

source ~/env/bin/activate

orion -vv hunt -n orion_sdbyol --config=orion_config.yaml \
  python3 ../../../main_pretrain.py \
    --dataset imagenet \
    --backbone resnet50 \
    --checkpoint_dir="$ROOT_PATH/{trial.hash_params}" \
    --data_dir $SLURM_TMPDIR/data \
    --train_dir $SLURM_TMPDIR/data/train/ \
    --val_dir   $SLURM_TMPDIR/data/val/ \
    --max_epochs 100 \
    --gpus 0,1 \
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
    --accumulate_grad_batches 16 \
    --classifier_lr 0.2 \
    --weight_decay 1e-6 \
    --batch_size 128 \
    --num_workers 4 \
    --dali \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --name "orion_sdbyol-{trial.hash_params}" \
    --entity il_group \
    --project VIL \
    --wandb \
    --save_checkpoint \
    --method sdbyol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --voc_size~'uniform(2,30,discrete=True)' \
    --message_size~'uniform(50,500,discrete=True)' \
    --tau_online~'uniform(0.1,10)' \
    --tau_target~'uniform(0.1,10)' \
    --momentum_classifier
