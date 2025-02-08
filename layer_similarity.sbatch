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

python layer_similarity.py --model_group medium