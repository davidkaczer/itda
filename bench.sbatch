#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos short
#SBATCH -t 02-00:00
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem 28G
#SBATCH -o logs/bench-%A_%a.out

################################################################################
# Usage:
#   sbatch bench.sbatch <RUN_ID> [EVAL_TYPES] [FORCE_RERUN_FLAG]
#
# Example:
#   sbatch bench.sbatch nx718rzm "core tpp" true
################################################################################

# 1) Parse arguments
RUN_ID=${1}
EVAL_TYPES=${2:-"core"}
FORCE_RERUN_FLAG=${3:-"false"}

# Build the --force_rerun option if requested
if [ "$FORCE_RERUN_FLAG" = "true" ]; then
    FORCE_RERUN="--force_rerun"
else
    FORCE_RERUN=""
fi

if [ -z "$RUN_ID" ]; then
  echo "Error: No run_id specified."
  echo "Usage: sbatch bench.sbatch <RUN_ID> [EVAL_TYPES] [FORCE_RERUN_FLAG]"
  exit 1
fi

# 2) Load modules and activate environment
module load cuda/11.3
source /home3/wclv88/venv/bin/activate

# 3) For logging
echo "==========================================="
echo "Run ID:          $RUN_ID"
echo "Eval types:      $EVAL_TYPES"
echo "Force re-run:    $FORCE_RERUN_FLAG"
echo "==========================================="

# 4) Execute the bench.py script
python3 bench.py \
    --run_id "$RUN_ID" \
    --eval_types $EVAL_TYPES \
    $FORCE_RERUN
