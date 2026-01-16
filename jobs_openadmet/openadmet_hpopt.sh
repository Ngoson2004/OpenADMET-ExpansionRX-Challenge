#!/bin/bash

#SBATCH --job-name=MTL_ExpansionRXONLY_3logits_Morgan_hpopt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256gb
#SBATCH --time=24:0:0
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm_out/slurm-%A.out
#SBATCH -A OD-231599
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngo014@csiro.au

. conda_virga.sh

cd model_scripts

start_time=$(date +%s)
python openadmet_hpopt.py \
        --conf best_config_auto_3logits.json \
        --result res_auto.csv \
        -ft morgan \
        -oo
end_time=$(date +%s)
elapsed=$((end_time - start_time))
 
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
total_minutes=$((elapsed / 60))

echo "Time taken for hp optimisation: ${hours}h ${minutes}m ${seconds}s"
echo "Total minutes: ${total_minutes} min"