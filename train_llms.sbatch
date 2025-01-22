#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos short
#SBATCH -t 02-00:00
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem 28G
#SBATCH --array=1-5
#SBATCH -o logs/train_llms-%A_%a.out

###############################################################################
# Usage:
#   sbatch train_llms.sbatch
#
# This will launch 5 parallel jobs (array indices 1..5). Each job:
#   1) Picks one seed from the SEEDS array below
#   2) Runs train_llms.py once with that seed
###############################################################################

# Define an array of seeds:
SEEDS=(42 1337 2023 9999 1234)

# Get the correct seed for this job
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID-1]}

# Print info for debugging/logging
echo "==========================================="
echo " SLURM_ARRAY_TASK_ID:   $SLURM_ARRAY_TASK_ID"
echo " Training seed:         $SEED"
echo "==========================================="

# Load modules and activate environment (adjust as needed)
module load cuda/11.3
source /home3/wclv88/venv/bin/activate

# Run the training script
python3 train_llms.py --seed "$SEED"
