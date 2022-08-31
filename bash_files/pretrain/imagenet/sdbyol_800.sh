#!/bin/bash

# ORION
export ORION_DB_TYPE=pickleddb
export ORION_DB_NAME=vil
export ORION_DB_ADDRESS=${1:-$PWD}/orion_db.pkl

# ENVIRONMENT
ROOT_PATH=$SCRATCH/sdbyol_search_IN_400
module load python/3.8
source ~/env_1.12/bin/activate

# LOADING DATA
../../../prepare_data.sh TEST

# RUNING SCRIPT
orion hunt -n orion_sdbyol_800 --config=orion_config.yaml \
  python3 ../../../main_pretrain.py \
    --devices 2 \
    --data_dir $SLURM_TMPDIR/data \
    --train_dir $SLURM_TMPDIR/data/train/ \
    --val_dir   $SLURM_TMPDIR/data/val/ \
    --checkpoint_dir=$ROOT_PATH \
    --entity il_group \
    --project sem_neurips \
    --dataset imagenet \
    --backbone resnet50 \
    --accumulate_grad_batches 1 \
    --name "sdbyol-ep:800" \
    --max_epochs 800 \
    --batch_size 32 \
    --tau_online~"uniform(0.1,10,precision=2)" \
    --tau_target~"uniform(0.1,10,precision=2)" \
    --voc_size 21 \
    --message_size 5000 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.2 \
    --classifier_lr 0.2 \
    --weight_decay 1.5e-6 \
    --num_workers 16 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --wandb \
    --offline \
    --save_checkpoint \
    --method sdbyol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.996 \
    --final_tau_momentum 1.0 \
    --auto_resume \
    --momentum_classifier

