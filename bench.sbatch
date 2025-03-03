#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos short
#SBATCH -t 02-00:00
#SBATCH --gres=gpu:ampere:1
#SBATCH -o logs/%x-%j.out
#SBATCH --mem 28G

module load cuda/11.3
source /home3/wclv88/venv/bin/activate

# Pass the run_id from the command line
RUN_ID=$1

python3 bench.py --run_id "$RUN_ID" --eval_types sparse_probing autointerp
