#!/bin/bash

#SBATCH --job-name=MTL_ExpansionRXONLY_NickHP_evidentialLoss_randomSplit_CDD_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256gb
#SBATCH --time=3:0:0
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm_out/slurm-%A.out
#SBATCH -A OD-231599
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngo014@csiro.au

. ./conda_virga.sh

cd model_scripts

start_time=$(date +%s)
python openadmet_train.py \
        --test-name openadmetONLY_NickHP_evidentialLoss_random_CDD_v2 \
        --conf best_config_Nick_wideFFN.json \
        --ckpt openadmetONLY_NickHP_evidentialLoss_random_CDD_v2 \
        -oo \
        -e 100 \
        --cdd \
        -ck SMILES
end_time=$(date +%s)
elapsed=$((end_time - start_time))
 
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
total_minutes=$((elapsed / 60))