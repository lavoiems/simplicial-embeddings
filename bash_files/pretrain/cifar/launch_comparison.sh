#!/bin/bash

# Create environment
module load python/3.8
virtualenv env

source env/bin/activate

pip install -r ../../../requirements.txt

# Launch jobs
sbatch --gres=gpu:1 -c 4 --mem=24G --array=1-5 ./simclr.sh
sbatch --gres=gpu:1 -c 4 --mem=24G --array=1-5 ./mocov2plus.sh
sbatch --gres=gpu:1 -c 4 --mem=24G --array=1-5 ./byol.sh
sbatch --gres=gpu:1 -c 4 --mem=24G --array=1-5 ./barlow.sh
sbatch --gres=gpu:1 -c 4 --mem=24G --array=1-5 ./swav.sh
sbatch --gres=gpu:1 -c 4 --mem=24G --array=1-5 ./dino.sh
sbatch --gres=gpu:1 -c 4 --mem=24G --array=1-5 ./vicreg.sh

sbatch --gres=gpu:1 -c 4 --mem=24G --array=1-5 ./sdsimclr.sh
sbatch --gres=gpu:1 -c 4 --mem=24G --array=1-5 ./sdmoco.sh
sbatch --gres=gpu:1 -c 4 --mem=24G --array=1-5 ./sdbyol.sh
sbatch --gres=gpu:1 -c 4 --mem=24G --array=1-5 ./sdbarlow.sh
sbatch --gres=gpu:1 -c 4 --mem=24G --array=1-5 ./sdswav.sh
sbatch --gres=gpu:1 -c 4 --mem=24G --array=1-5 ./sddino.sh
sbatch --gres=gpu:1 -c 4 --mem=24G --array=1-5 ./sdvicreg.sh

