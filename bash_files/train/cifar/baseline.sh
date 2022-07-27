#!/bin/bash
ROOT_PATH=$1
BACKBONE=${2:-resnet50}
DATASET=${4:-cifar100}
GROUP="${DATASET}_${BACKBONE}_sgd"
OGROUP="classify_${GROUP}"
EXPNAME="${OGROUP}_{trial.hash_params}"
DATA_PATH=${SLURM_TMPDIR}/data

. ../../../start.sh pickleddb "${ROOT_PATH}/${OGROUP}"

orion -v hunt -n ${OGROUP} --config=../../../orion_config.yaml \
  python ../../../main_train.py \
  --wandb \
  --seed~'choices([0,1,2,3,5])' \
  --entity il_group \
  --project sem_classification \
  --name $EXPNAME \
  --group $OGROUP \
  --dataset ${DATASET} \
  --backbone ${BACKBONE} \
  --data_dir ${DATA_PATH}/${DATASET} \
  --train_dir ${DATA_PATH}/${DATASET} \
  --val_dir ${DATA_PATH}/${DATASET} \
  --checkpoint_dir="${ROOT_PATH}/${OGROUP}/${EXPNAME}" \
  --num_workers=${SLURM_CPUS_PER_TASK} \
  --save_checkpoint \
  --auto_resume \
  --max_epochs 500 \
  --gpus 0 \
  --accelerator gpu \
  --precision 16 \
  --optimizer sgd \
  --lars \
  --grad_clip_lars \
  --eta_lars 0.02 \
  --exclude_bias_n_norm \
  --scheduler warmup_cosine \
  --lr 1.0 \
  --weight_decay 1e-5 \
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
