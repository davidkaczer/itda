#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos short
#SBATCH -t 02-00:00
#SBATCH --gres=gpu:ampere:1
#SBATCH -o logs/qwen_sae-%A_%a.out
#SBATCH --mem 28G

# Get the script arguments from the `sbatch` command
MODEL=${1:-EleutherAI/pythia-70m-deduped}
BATCH_SIZE=${2:-32}
LAYER=${3:-3}
TARGET_LOSS=${4:-0.01}
L0=${5:-40}

# Print the parameters for logging
echo "Running training with the following parameters:"
echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Layer: $LAYER"
echo "Target loss: $TARGET_LOSS"
echo "L0: $L0"

# Run the Python script with the provided parameters
python3 train.py \
    --model "$MODEL" \
    --batch_size "$BATCH_SIZE" \
    --layer "$LAYER" \
    --target_loss "$TARGET_LOSS" \
    --l0 "$L0"
