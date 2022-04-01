#!/bin/bash

VIL_PASS=RSFTHfHoHYWUWx5m
export ORION_DB_TYPE=mongodb
export ORION_DB_NAME=vil
if [[ $beluga == 0 ]] || [[ $graham == 0 ]] || [[ $narval == 0 ]]; then
  if [ $beluga == 0 ]; then
    remote_host=beluga$(((RANDOM%6)+1)).int.ets1.calculquebec.ca
  elif [ $graham == 0 ];
  then
    remote_host=gra-login$(((RANDOM%3)+1))
  elif [ $narval == 0 ]; then
    remote_host=narval$(((RANDOM%3)+1))
  else
    echo "assertion unreachable"
    exit -1
  fi
  echo "Access MongoDB with SSH tunnel"
  export ORION_DB_ADDRESS="mongodb://lavoiems:$VIL_PASS@localhost/lavoiems?authSource=lavoiems&ssl_match_hostname=false"
  export ORION_DB_PORT=$(python -c "from socket import socket; s = socket(); s.bind((\"\", 0)); print(s.getsockname()[1])")
  ssh -o StrictHostKeyChecking=no $remote_host -L $ORION_DB_PORT:34.195.91.43:27017 -n -N -f
else
  echo "Access MongoDB"
  export ORION_DB_ADDRESS="mongodb://lavoiems:$VIL_PASS@34.195.91.43/lavoiems?authSource=lavoiems"
fi

ROOT_PATH=$1

#../../../prepare_data.sh TEST

module load python/3.8
source ~/env/bin/activate

orion --debug hunt -n orion_sdmoco_y --config=orion_config.yaml \
  python3 ../../../main_pretrain.py \
    --dataset imagenet \
    --backbone resnet50 \
    --checkpoint_dir="$ROOT_PATH/{trial.hash_params}" \
    --data_dir $SLURM_TMPDIR/data \
    --train_dir $SLURM_TMPDIR/data/train/ \
    --val_dir   $SLURM_TMPDIR/data/val/ \
    --max_epochs 100 \
    --gpus 0 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --classifier_lr 0.4 \
    --weight_decay 3e-5 \
    --batch_size 128 \
    --num_workers 5 \
    --dali \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --name "sdmoco-{trial.hash_params}" \
    --entity il_group \
    --project VIL \
    --wandb \
    --save_checkpoint \
    --method sdmoco \
    --proj_hidden_dim 2048 \
    --queue_size 65536 \
    --temperature 0.1 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 0.999 \
    --voc_size=21 \
    --message_size=4500 \
    --tau_online~"loguniform(0.1,5,precision=3)" \
    --tau_target~"loguniform(0.1,5,precision=3)" \
    --momentum_classifier
