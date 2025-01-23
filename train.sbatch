#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos short
#SBATCH -t 02-00:00
#SBATCH --gres=gpu:pascal:1
#SBATCH -o logs/qwen_sae-%A_%a.out
#SBATCH --mem 28G

# -----------------------------------------------------------------------------
# Slurm job script for training using train.py
# -----------------------------------------------------------------------------
# This script supports both ITO-based and dictionary-learning-based training.
# It accepts the following arguments (in order):
#
# 1) METHOD        (ito or dictlearn) [default: ito]
# 2) MODEL         [default: EleutherAI/pythia-70m-deduped]
# 3) DATASET       [default: NeelNanda/pile-10k]
# 4) LAYER         [default: 3]
# 5) BATCH_SIZE    [default: 32]
# 6) SEQ_LEN       [default: 128]
# 7) MAX_SEQUENCES [default: None — i.e., no limit]
# 8) L0            [default: 40 for ITO]
# 9) TARGET_LOSS   [default: 3.0 for ITO]
# 10) LOAD_RUN_ID  [default: None — for continuing a previous ITO run]
# 11) DICT_SIZE    [default: 65536 for dictlearn]
# 12) LR           [default: 1e-3 for dictlearn]
# -----------------------------------------------------------------------------

# Grab arguments from command line; if not provided, use the defaults:
METHOD=${1:-ito}               # "ito" or "dictlearn"
MODEL=${2:-EleutherAI/pythia-70m-deduped}
DATASET=${3:-NeelNanda/pile-10k}
LAYER=${4:-3}
BATCH_SIZE=${5:-32}
SEQ_LEN=${6:-128}

# For MAX_SEQUENCES, we accept an integer or "None" as the default:
MAX_SEQUENCES=${7:-None}

# ITO-related
L0=${8:-40}
TARGET_LOSS=${9:-3.0}
LOAD_RUN_ID=${10:-None}

# Dictionary-learning–related
DICT_SIZE=${11:-65536}
LR=${12:-1e-3}

# -----------------------------------------------------------------------------
# Print parameters for logging
# -----------------------------------------------------------------------------
echo "Running training with the following parameters:"
echo "METHOD:        $METHOD"
echo "MODEL:         $MODEL"
echo "DATASET:       $DATASET"
echo "LAYER:         $LAYER"
echo "BATCH_SIZE:    $BATCH_SIZE"
echo "SEQ_LEN:       $SEQ_LEN"
echo "MAX_SEQUENCES: $MAX_SEQUENCES"
echo "L0:            $L0"
echo "TARGET_LOSS:   $TARGET_LOSS"
echo "LOAD_RUN_ID:   $LOAD_RUN_ID"
echo "DICT_SIZE:     $DICT_SIZE"
echo "LR:            $LR"

module load cuda/11.3
source /home3/wclv88/venv/bin/activate

# -----------------------------------------------------------------------------
# Execute training
# -----------------------------------------------------------------------------
python3 train.py \
    --method "$METHOD" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --layer "$LAYER" \
    --batch_size "$BATCH_SIZE" \
    --seq_len "$SEQ_LEN" \
    --max_sequences "$MAX_SEQUENCES" \
    --l0 "$L0" \
    --target_loss "$TARGET_LOSS" \
    --load_run_id "$LOAD_RUN_ID" \
    --dict_size "$DICT_SIZE" \
    --lr "$LR"

