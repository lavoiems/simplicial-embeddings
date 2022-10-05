#!/bin/bash
ROOT_PATH=$1
SEED=${2:-0}
DATASET=${3:-cifar100}
BACKBONE=${4:-resnet18}
METHOD=${5:-vqbyol}
GROUP="${METHOD}_${DATASET}_${BACKBONE}"
OGROUP="search_soft_mby_spherical_${GROUP}"
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
  --max_epochs~'fidelity(200,1000,base=4)' \
  --total_max_epochs 1000 \
  --model_selection_score 'val/online_y_acc1' \
  --gpus 0 \
  --accelerator gpu \
  --precision 16 \
  --optimizer sgd \
  --lars \
  --grad_clip_lars \
  --eta_lars 0.02 \
  --exclude_bias_n_norm \
  --scheduler warmup_cosine \
  --lr~'uniform(0.2,0.5,precision=1)' \
  --classifier_lr 0.1 \
  --classifier_wd 1e-5 \
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
  --voc_size~'choices([8,16,32])' \
  --message_size~'choices([512,1024,2048,4096])' \
  --shared_codebook True \
  --vq_code_wd 1e-5 \
  --vq_dimz 8 \
  --vq_rel_loss~'loguniform(1e-5,1e-3,precision=1)' \
  --vq_update_rule=loss \
  --vq_hard False \
  --vq_ladder False \
  --vq_tau~'choices([0.05,0.1,0.2,0.3,0.4,0.5])' \
  --vq_spherical~'choices([True,False])'

# EXPNAME=search2_soft_no_ladder_vqbyol_cifar100_resnet18_832fc7d6e05422c66d9545ad61119301
# EXPNAME=search2_soft_no_ladder_vqbyol_cifar100_resnet18_5d2a533edef06f47f2cc5eaf036cffb2
# EXPNAME=search2_soft_no_ladder_vqbyol_cifar100_resnet18_86fd56ba3a41b1dfbf5d9f72dcde90bf
# EXPNAME=search2_soft_no_ladder_vqbyol_cifar100_resnet18_dd89de0a79dac2e0158593bad1dcc985
# EXPNAME=search2_soft_no_ladder_vqbyol_cifar100_resnet18_b26c75f4ca5944eb7fe3a0ffff6a3b39
# EXPNAME=search2_soft_no_ladder_vqbyol_cifar100_resnet18_14fd3b5987e968fe534678b47bbc4768
EXPNAME=search2_soft_no_ladder_vqbyol_cifar100_resnet18_ec5ee6631e4ef2eb547acd896f764b73
EXPDIR="/network/scratch/t/tsirigoc/sdbyol_iclr23/vqbyol_cifar100_resnet18/${EXPNAME}"

orion -v hunt -n ${EXPNAME}_linear --config=../../../orion_config.yaml \
  python3 /home/mila/t/tsirigoc/solo-learn/main_linear.py \
  --wandb \
  --seed 0 \
  --name $EXPNAME \
  --method vqbyol --dataset cifar100 \
  --data_dir /home/mila/t/tsirigoc/datasets \
  --checkpoint_dir "${EXPDIR}/linear/linear_{trial.hash_params}" \
  --save_checkpoint --auto_resume --num_workers ${SLURM_CPUS_PER_TASK} \
  --model_selection_score "val/y_acc1" \
  --max_epochs 100 --total_max_epochs 100 \
  --gpus 0 --accelerator gpu --precision 16 \
  --optimizer sgd --scheduler step --lr_decay_steps 60 80 \
  --lr~'choices([0.01,0.02,0.05,0.1,0.2,0.5])' \
  --weight_decay~'choices([0.0,1e-6,1e-5,1e-4])' \
  --l1~'choices([0.0,1e-6,1e-5,1e-4])' \
  --pretrain_augs~'choices([True,False])' \
  --batch_size 256 --crop_size 32 \
  --pretrained_feature_extractor "${EXPDIR}/last.ckpt"
