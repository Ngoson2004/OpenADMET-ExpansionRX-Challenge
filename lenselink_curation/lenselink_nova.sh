#!/bin/bash

#SBATCH --job-name=lenselink_curation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:0:0
#SBATCH --output=./slurm_out/slurm-%A.out
#SBATCH -A OD-231599
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngo014@csiro.au

. ../conda_cmd.sh
source activate chemprop-py311

start_time_L=$(date +%s)

python lenselink_curation.py \
    -i ../data/Nova274K_ultimate.csv ../data/Nova16k_ultimate.csv \
    -o ../data/LenseLink_data/LenseLink_Novartis.csv \
    --transf-default x \
    --aggregation mean \
    --keep-only-numeric \

end_time_L=$(date +%s)

elapsed_L=$((end_time_L - start_time_L))

total_minutes_L=$((elapsed_L / 60))

 echo "The whole process takes ${total_minutes_L} minutes"

#../data/Nova16k_train.csv \
# --value-col Caco-2_LogPapp LE-MDCKv2_LogPapp LE-MDCKv1_LogPapp logPAMPA rLM LogCLint hLM LogCLint mLM LogCLint minipigLM LogCLint cynoLM LogCLint dLM LogCLint \
# mod1 mod2 mod3 mod4