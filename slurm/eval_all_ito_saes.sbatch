#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos short
#SBATCH -t 01-00:00
#SBATCH --gres=gpu
#SBATCH -o logs/eval_all_ito_saes-%A_%a.out
#SBATCH --mem 8G
#SBATCH --array=0-15

L0s=(4 8 16 32)
LOSSES=(1.5 2.0 2.5 3.0)

INDEX=${SLURM_ARRAY_TASK_ID}
L0_INDEX=$((INDEX / 4))
LOSS_INDEX=$((INDEX % 4))

L0=${L0s[$L0_INDEX]}
TARGET_LOSS=${LOSSES[$LOSS_INDEX]}

echo "Evaluating ITO SAE with L0=$L0 and TARGET_LOSS=$TARGET_LOSS"

module load cuda/11.3 # adjust if needed
source /path/to/your/venv/bin/activate

# Use the filter_runs function within `train_ito_sae.py` or a separate script
# that you have written to find the run. Suppose `train_ito_sae.py` has a
# `--filter_runs` mode or we can quickly write a small snippet here.

# We'll assume `train_ito_sae.py` can be run with a special flag to print run_ids
# matching given arguments. For example:
# python train_ito_sae.py --filter_runs --model gpt2 --layer 8 --l0 ${L0} --target_loss ${TARGET_LOSS}

RUN_ID=$(python train_ito_sae.py --filter_runs --model gpt2 --layer 8 --l0 ${L0} --target_loss ${TARGET_LOSS})

if [ -z "$RUN_ID" ]; then
    echo "No run found for these parameters. Ensure training was completed."
    exit 1
fi

echo "Found run_id: $RUN_ID"

# Now run the benchmarking code on this run_id
python benchmark_sae.py \
  --model gpt2 \
  --ito_run_id $RUN_ID \
  --eval_types core scr tpp sparse_probing
