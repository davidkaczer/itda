#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos short
#SBATCH -t 02-00:00
#SBATCH --gres=gpu:pascal:1
#SBATCH -o logs/qwen_sae-%A_%a.out
#SBATCH --mem 28G
#SBATCH --array=0-47

module load cuda/11.3
source /home3/wclv88/venv/bin/activate

MODELS=(
  "Qwen/Qwen2-0.5B-Instruct"
  "Qwen/Qwen2-0.5B"
)

INDEX=$SLURM_ARRAY_TASK_ID
MODEL_INDEX=$((INDEX / 24))
LAYER=$((INDEX % 24))
MODEL=${MODELS[$MODEL_INDEX]}

if [ "$INDEX" -eq 0 ]; then
  BATCH_SIZE=128
else
  BATCH_SIZE=1024
fi

echo "Running command for model: $MODEL, layer: $LAYER, batch_size: $BATCH_SIZE"
python3 train.py \
  --model "$MODEL" \
  --batch_size "$BATCH_SIZE" \
  --target_dict_size 50000 \
  --hook_name "blocks.$LAYER.hook_resid_pre"
