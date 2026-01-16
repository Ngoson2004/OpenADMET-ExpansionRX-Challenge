#!/bin/bash

#SBATCH --job-name=MTL_ExpansionRXONLY_submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:0:0
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm_out/slurm-%A.out
#SBATCH -A OD-231599
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngo014@csiro.au

. ./conda_virga.sh

cd model_scripts

python openadmet_submission.py \
        -n Novartis_Lenselink55_AZ_QM9_concat_NickHP_evidentialLoss_random_gradClip_10epochs \
        -m "ensemble models" \
        -o MEGA_submission_v2.csv
        # -d maplight