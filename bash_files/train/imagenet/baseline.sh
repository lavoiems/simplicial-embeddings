#!/bin/bash
ROOT_PATH=$1
BACKBONE=${2:-resnet50}
DATASET=${4:-imagenet}
GROUP="${DATASET}_${BACKBONE}_sgd"
OGROUP="classify_${GROUP}"
EXPNAME="${OGROUP}_{trial.hash_params}"
DATA_PATH=${SLURM_TMPDIR}/data

. ../../../start.sh pickleddb "${ROOT_PATH}/${OGROUP}"

../../../prepare_data.sh TEST

orion -v hunt -n ${OGROUP} --config=../../../orion_config.yaml \
  python ../../../main_train.py \
  --wandb \
  --seed=0 \
  --entity il_group \
  --project sem_classification \
  --name $EXPNAME \
  --group $OGROUP \
  --dataset ${DATASET} \
  --backbone ${BACKBONE} \
  --data_dir $SLURM_TMPDIR \
  --train_dir $SLURM_TMPDIR/data/train \
  --val_dir   $SLURM_TMPDIR/data/val \
  --checkpoint_dir="${ROOT_PATH}/${OGROUP}/${EXPNAME}" \
  --num_workers=${SLURM_CPUS_PER_TASK} \
  --save_checkpoint \
  --auto_resume \
  --max_epochs 200 \
  --gpus 0,1,2,3 \
  --accelerator gpu \
  --strategy ddp \
  --sync_batchnorm \
  --dali \
  --precision 16 \
  --optimizer sgd \
  --lars \
  --eta_lars 0.001 \
  --exclude_bias_n_norm \
  --scheduler warmup_cosine \
  --lr 0.45 \
  --weight_decay 1e-6 \
  --batch_size 256 \
  --use_sem~'choices([True,False])' \
  --message_size 5000 \
  --voc_size 13 \
  --tau 1. \
  --brightness 0.4 \
  --contrast 0.4 \
  --saturation 0.2 \
  --hue 0.1 \
  --gaussian_prob 0.1 \
  --solarization_prob 0.2 \
  --pretrain_augs True
