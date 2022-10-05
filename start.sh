#!/bin/bash

echo $SLURM_TMPDIR
echo $HOSTNAME
echo $SLURM_CLUSTER_NAME
echo $(date)

[[ $SLURM_CLUSTER_NAME == beluga ]]; beluga=$?;
[[ $SLURM_CLUSTER_NAME == graham ]]; graham=$?;
[[ $SLURM_CLUSTER_NAME == narval ]]; narval=$?;
[[ $SLURM_CLUSTER_NAME == mila ]]; mila=$?;

if [ $narval == 0 ]; then
module load StdEnv/2020 intel/2020.1.217
fi

module load python/3.8

if [ $mila == 0 ]; then
echo "Loading Mila modules"
module load gcc/8.4.0 cuda/11.2/cudnn
module load cuda/11.2/nccl
module load cudatoolkit/11.2
else
module load cuda/11.4
module load cudnn 2> /dev/null || module load cuda/11.4/cudnn
fi

nvidia-smi


ENVDIR="$SLURM_TMPDIR/env"

# virtualenv --no-download $ENVDIR

if [ $mila == 0 ]; then
module unload python/3.8
fi

source $HOME/env/devel/bin/activate
# source $ENVDIR/bin/activate

# if [ $mila == 0 ];
# then
#   echo "In Mila - $mila"
#   # pip install --upgrade pip wheel setuptools
#   pip install -r ../../../requirements.txt
#   pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# #   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda102
# #   # pip install onnxruntime_training==1.11.0 -f https://download.onnxruntime.ai/onnxruntime_stable_cu102.html
# #   # pip install torch_ort
# #   # python -m torch_ort.configure
# else
#   echo "In CC - $mila"
#   pip install --no-index --upgrade pip wheel setuptools
#   pip install --no-index -r ../../../requirements.txt
# fi

DB_TYPE=${1:-mongodb}
export ORION_DB_TYPE=$DB_TYPE
if [[ $DB_TYPE == "pickleddb" ]]; then
  export ORION_DB_ADDRESS=${2:-$PWD}/orion_db.pkl
  echo "Orion using pickleddb at: ${ORION_DB_ADDRESS}"
elif [[ $DB_TYPE == "mongodb" ]]; then
  VIL_PASS=RSFTHfHoHYWUWx5m
  export ORION_DB_NAME=vil
  if [[ $beluga == 0 ]] || [[ $graham == 0 ]] || [[ $narval == 0 ]]; then
    if [ $beluga == 0 ]; then
      remote_host=beluga$(((RANDOM%6)+1)).int.ets1.calculquebec.ca
    elif [ $graham == 0 ]; then
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

  orion db test
  echo "Orion using mongodb at: ${ORION_DB_ADDRESS}"
else
  echo "No Orion will be used..."
fi
