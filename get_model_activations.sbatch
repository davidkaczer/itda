#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos short
#SBATCH -t 02-00:00
#SBATCH --gres=gpu:pascal:1
#SBATCH -o logs/activations-%A_%a.out
#SBATCH --mem 28G

# Get the script arguments from the `sbatch` command
BATCH_SIZE=${1:-16}
MODEL=${2:-google/gemma-2-2}

# Print the parameters for logging
echo "Running model activations with the following parameters:"
echo "Batch size: $BATCH_SIZE"
echo "Model: $MODEL"

# Run the Python script with the provided parameters
python3 get_model_activations.py \
    --batch_size "$BATCH_SIZE" \
    --model "$MODEL"
