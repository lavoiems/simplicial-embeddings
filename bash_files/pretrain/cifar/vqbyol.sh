#!/bin/bash
ROOT_PATH=$1
SEED=${2:-0}
DATASET=${3:-cifar100}
BACKBONE=${4:-resnet18}
METHOD=${5:-vqbyol}
GROUP="${METHOD}_${DATASET}_${BACKBONE}"
OGROUP="search_${GROUP}"
EXPNAME="${OGROUP}_{trial.hash_params}"
# EXPNAME="${OGROUP}_${SEED}"
DATA_PATH=${SLURM_TMPDIR}/data

. ../../../start.sh pickleddb "${ROOT_PATH}/${OGROUP}"
# . ../../../start.sh noorion "${ROOT_PATH}/${OGROUP}"

export WANDB_ENTITY=tsirif
export WANDB_PROJECT=sdbyol
export WANDB_RUN_GROUP=$OGROUP

if [[ $DATASET == "mpi3d" ]]; then
  TRAIN_DATA_PATH=$HOME/datasets/mpi3d_real_train_3_50000.npz
  VAL_DATA_PATH="$HOME/datasets/mpi3d_real_iid_valid_3_10000.npz $HOME/datasets/mpi3d_real_ood_valid_3_10000.npz"
  # VAL_DATA_PATH="$HOME/datasets/mpi3d_real_iid_valid_3_10000.npz"
  CROP_SIZE=64
  BRIGHTNESS=0.4
  CONTRAST=0.4
  SATURATION=0.2
  HUE=0.1
  MINSCALE=0.2
  HOR_FLIP=0.
else
  TRAIN_DATA_PATH=$HOME/datasets
  VAL_DATA_PATH=$HOME/datasets
  CROP_SIZE=32
  BRIGHTNESS=0.4
  CONTRAST=0.4
  SATURATION=0.2
  HUE=0.1
  MINSCALE=0.08
  HOR_FLIP=0.5
fi

orion -v hunt -n ${OGROUP} --config=../../../orion_config.yaml \
  python3 ../../../main_pretrain.py \
  --wandb \
  --seed $SEED \
  --name $EXPNAME \
  --method $METHOD \
  --dataset ${DATASET} \
  --backbone ${BACKBONE} \
  --data_dir $TRAIN_DATA_PATH \
  --checkpoint_dir="${ROOT_PATH}/${GROUP}/${EXPNAME}" \
  --save_checkpoint \
  --auto_resume \
  --num_workers ${SLURM_CPUS_PER_TASK} \
  --max_epochs~'fidelity(10,1000,base=4)' \
  --total_max_epochs 1000 \
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
  --classifier_lr 0.1 \
  --weight_decay 1e-5 \
  --batch_size 256 \
  --brightness 0.4 \
  --contrast 0.4 \
  --saturation 0.2 \
  --hue 0.1 \
  --gaussian_prob 0.0 0.0 \
  --solarization_prob 0.0 0.2 \
  --crop_size 32 \
  --num_crops_per_aug 1 1 \
  --proj_output_dim 256 \
  --proj_hidden_dim 4096 \
  --pred_hidden_dim 4096 \
  --base_tau_momentum 0.99 \
  --final_tau_momentum 1.0 \
  --momentum_classifier \
  --voc_size~'choices([128,256,512])' \
  --message_size~'choices([128,256,512])' \
  --shared_codebook~'choices([True,False])' \
  --vq_code_wd~'choices([0.,1e-5])' \
  --vq_dimz~'choices([16,32])' \
  --vq_rel_loss~'loguniform(0.001,10.,precision=2)' \
  --vq_update_rule=loss

