#!/bin/bash
ROOT_PATH=$1
METHOD=${2:-sdsimclr}
DATASET=${3:-cifar100}
BACKBONE=${4:-resnet50}
OPTIM=${5:-sgd}
GROUP="${METHOD}_${DATASET}_${BACKBONE}_${OPTIM}"
OGROUP="benchmark_${GROUP}"
EXPNAME="${OGROUP}_{trial.hash_params}"
DATA_PATH=${SLURM_TMPDIR}/data

. ../../../start.sh pickleddb "${ROOT_PATH}/${OGROUP}"

orion -v hunt -n ${OGROUP} --config=../../../orion_config.yaml \
  python3 ../../../main_pretrain.py \
  --seed=0 \
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
  --lr 0.4 \
  --classifier_lr 0.1 \
  --weight_decay 1e-5 \
  --batch_size 256 \
  --brightness 0.8 \
  --contrast 0.8 \
  --saturation 0.8 \
  --hue 0.2 \
  --gaussian_prob 0.0 0.0 \
  --crop_size 32 \
  --num_crops_per_aug 1 1 \
  --temperature 0.2 \
  --proj_hidden_dim 2048 \
  --proj_output_dim 256 \
  --voc_size 13 \
  --message_size 5000 \
  --tau_online~'uniform(0.1,2.,precision=2)' \
  --tau_target~'uniform(0.1,2.,precision=2)'
