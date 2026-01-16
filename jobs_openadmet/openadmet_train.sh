#!/bin/bash

#SBATCH --job-name=MTL_Lenselink55_ExpansionRX_Nickhp_gradClip_training_10epochs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256gb
#SBATCH --time=8:0:0
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm_out/slurm-%A.out
#SBATCH -A OD-231599
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngo014@csiro.au

. ./conda_virga.sh

cd model_scripts

start_time=$(date +%s)
python openadmet_train.py \
        --test-name lenselink55_NickHP_evidentialLoss_random_gradClip_10epochs \
        --conf best_config_Nick_wideFFN.json \
        --ckpt lenselink55_NickHP_evidentialLoss_random_gradClip_10epochs \
        -e 10 \
        -us \
        -gc 1 
        # --cdd
        # -ft maplight
end_time=$(date +%s)
elapsed=$((end_time - start_time))
 
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
total_minutes=$((elapsed / 60))