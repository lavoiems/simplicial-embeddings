#!/bin/bash
ROOT_PATH=$1
SEED=${2:-0}
DATASET=${3:-cifar100}
BACKBONE=${4:-resnet18}
METHOD=${5:-sdbyol}
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
  python ../../../main_pretrain.py \
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
  --max_epochs 1000 \
  --total_max_epochs 1000 \
  --model_selection_score 'val/online_y_1.0_acc1' \
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
  --voc_size~'choices([8,12,16,24,32])' \
  --message_size~'choices([512,1024,2048,4096])' \
  --taus 0.1 0.5 1.0 \
  --tau_online 1.0 \
  --tau_target 1.0


# EXPNAME=search_sdbyol_cifar100_resnet18_1905ec5763468a0f2af9764e55402da4
# EXPDIR="/network/scratch/t/tsirigoc/sdbyol_iclr23/sdbyol_cifar100_resnet18/${EXPNAME}"

# orion -v hunt -n ${EXPNAME}_linear --config=../../../orion_config.yaml \
# python3 /home/mila/t/tsirigoc/solo-learn/main_linear.py \
#   --wandb \
#   --seed 0 \
#   --name ${EXPNAME}_2 \
#   --method sdbyol --dataset cifar100 --linear_method linear_control \
#   --data_dir /home/mila/t/tsirigoc/datasets \
#   --checkpoint_dir "${EXPDIR}/linear2" \
#   --save_checkpoint --auto_resume --num_workers ${SLURM_CPUS_PER_TASK} \
#   --max_epochs 100 \
#   --gpus 0 --accelerator gpu --precision 16 \
#   --optimizer sgd --scheduler step --lr_decay_steps 60 80 \
#   --voc_size 32 --message_size 2048 \
#   --lrs 0.3 0.4 0.5 \
#   --wd1 0 1e-5 \
#   --wd2 0 1e-5 \
#   --taus 0.1 1 2 \
#   --eval_taus 0.1 1 2 \
#   --class_base True \
#   --pretrain_augs False \
#   --batch_size 256 --crop_size 32 \
#   --pretrained_feature_extractor "${EXPDIR}/last.ckpt"
