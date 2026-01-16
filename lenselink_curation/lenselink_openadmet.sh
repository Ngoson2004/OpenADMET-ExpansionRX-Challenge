#!/bin/bash

#SBATCH --job-name=lenselink_openADMET_curation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:0:0
#SBATCH --output=./slurm_out/slurm-%A.out
#SBATCH -A OD-231599
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngo014@csiro.au

. ../conda_virga.sh

start_time=$(date +%s)
python lenselink_curation.py \
    -i ../data/open_admet/openadmet_train.csv \
    -o  ../data/open_admet/curated_openadmet_train.csv \
    --keep-only-numeric \
    --transf-dict '{"gmb":"log10(x/(100-x))", "log" : "x"}' \
    --transf-default "log10(x)"
end_time=$(date +%s)
elapsed=$((end_time - start_time))
total_minutes=$((elapsed / 60))
echo "Chemistry curation takes ${total_minutes} minutes"