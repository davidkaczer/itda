#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos short
#SBATCH -t 02-00:00
#SBATCH --gres=gpu:ampere:1
#SBATCH -o logs/train_all_ito_saes-%A_%a.out
#SBATCH --mem 28G
#SBATCH --array=0-0

# Array of L0 values and target losses
L0s=(4 8 16 32)
LOSSES=(2.0 2.5 3.0)

# Determine indices from the job array task ID
INDEX=${SLURM_ARRAY_TASK_ID}
L0_INDEX=$((INDEX / 4))
LOSS_INDEX=$((INDEX % 4))

L0=${L0s[$L0_INDEX]}
TARGET_LOSS=${LOSSES[$LOSS_INDEX]}

echo "Training ITO SAE with L0=$L0 and TARGET_LOSS=$TARGET_LOSS"

module load cuda/11.3
source /home3/wclv88/venv/bin/activate

python ../ito.py \
  --model gpt2 \
  --layer 8 \
  --l0 ${L0} \
  --target_loss ${TARGET_LOSS} \
  --seq_len 128 \
  --batch_size 256

