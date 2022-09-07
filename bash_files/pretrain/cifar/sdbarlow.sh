#!/bin/bash
ROOT_PATH=$1
SEED=${2:-0}
METHOD=${3:-sdbt}
DATASET=${4:-cifar100}
BACKBONE=${5:-resnet50}
OPTIM=${6:-sgd}
GROUP="${METHOD}_${DATASET}_${BACKBONE}_${OPTIM}"
OGROUP="benchmark_${GROUP}"
# EXPNAME="${OGROUP}_{trial.hash_params}"
EXPNAME="${OGROUP}_${SEED}"
DATA_PATH=${SLURM_TMPDIR}/data

# . ../../../start.sh pickleddb "${ROOT_PATH}/${OGROUP}"
. ../../../start.sh noorion "${ROOT_PATH}/${OGROUP}"

# orion -v hunt -n ${OGROUP} --config=../../../orion_config.yaml \
python3 ../../../main_pretrain.py \
  --seed $SEED \
  --wandb \
  --entity il_group \
  --project sem_neurips \
  --name $EXPNAME \
  --group $OGROUP \
  --method $METHOD \
  --dataset ${DATASET} \
  --backbone ${BACKBONE} \
  --data_dir ${DATA_PATH}/${DATASET} \
  --train_dir ${DATA_PATH}/${DATASET} \
  --val_dir ${DATA_PATH}/${DATASET} \
  --checkpoint_dir="${ROOT_PATH}/${OGROUP}/${EXPNAME}" \
  --num_workers=${SLURM_CPUS_PER_TASK} \
  --save_checkpoint \
  --auto_resume \
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
  --lr 0.3 \
  --weight_decay 1e-4 \
  --batch_size 256 \
  --brightness 0.4 \
  --contrast 0.4 \
  --saturation 0.2 \
  --hue 0.1 \
  --gaussian_prob 0.0 0.0 \
  --solarization_prob 0.0 0.2 \
  --crop_size 32 \
  --num_crops_per_aug 1 1 \
  --proj_hidden_dim 2048 \
  --proj_output_dim 2048 \
  --scale_loss 0.1 \
  --voc_size 13 \
  --message_size 5000 \
  --tau_online 1. \
  --tau_target 0.99 \
  --taus 0.1 1. 2.
