#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos short
#SBATCH -t 02-00:00
#SBATCH --gres=gpu:pascal:1
#SBATCH -o logs/activations-%A_%a.out
#SBATCH --mem 28G

# Positional arguments with defaults
BATCH_SIZE=${1:-32}
MODEL=${2:-"gpt2"}
DATASET=${3:-"NeelNanda/pile-10k"}
SEQ_LEN=${4:-128}
ACTIVATIONS_PATH=${5:-"artifacts/data"}
NUM_EXAMPLES=${6:-10000}
STOP_FORWARD_PASS_LAYER=${7}      # optional
OFFLOAD_FOLDER=${8}               # optional
LAYERS=${9}                       # optional
REVISION=${10}                    # optional

module load cuda/11.3
source /home3/wclv88/venv/bin/activate

# Print the parameters for logging
echo "Running activation collection with the following parameters:"
echo "  BATCH_SIZE:             $BATCH_SIZE"
echo "  MODEL:                  $MODEL"
echo "  DATASET:                $DATASET"
echo "  SEQ_LEN:                $SEQ_LEN"
echo "  ACTIVATIONS_PATH:       $ACTIVATIONS_PATH"
echo "  NUM_EXAMPLES:           $NUM_EXAMPLES"
echo "  STOP_FORWARD_PASS_LAYER: $STOP_FORWARD_PASS_LAYER"
echo "  OFFLOAD_FOLDER:         $OFFLOAD_FOLDER"
echo "  LAYERS:                 $LAYERS"
echo "  REVISION:               $REVISION"

# Build up the command with mandatory arguments
CMD="python3 get_model_activations.py \
  --batch_size $BATCH_SIZE \
  --model \"$MODEL\" \
  --dataset \"$DATASET\" \
  --seq_len $SEQ_LEN \
  --activations_path \"$ACTIVATIONS_PATH\" \
  --num_examples $NUM_EXAMPLES"

# Conditionally add optional arguments if they were provided
if [ -n "$STOP_FORWARD_PASS_LAYER" ]; then
  CMD="$CMD --stop_forward_pass_layer $STOP_FORWARD_PASS_LAYER"
fi

if [ -n "$OFFLOAD_FOLDER" ]; then
  CMD="$CMD --offload_folder \"$OFFLOAD_FOLDER\""
fi

if [ -n "$LAYERS" ]; then
  CMD="$CMD --layers \"$LAYERS\""
fi

if [ -n "$REVISION" ]; then
  CMD="$CMD --revision \"$REVISION\""
fi

echo "Final command:"
echo "$CMD"
echo "--------------------------------"

# Execute the command
eval $CMD
