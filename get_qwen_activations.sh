#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos short
#SBATCH -t 02-00:00
#SBATCH --gres=gpu:ampere:1
#SBATCH -o logs/qwen_sae-%A_%a.out
#SBATCH --mem 28G

module load cuda/11.3
source /home3/wclv88/venv/bin/activate

python3 get_model_activations.py --batch_size 512 --model Qwen/Qwen2-0.5B  --hook_name "blocks.*.hook_resid_pre"
python3 get_model_activations.py --batch_size 512 --model Qwen/Qwen2-0.5B-Instruct  --hook_name "blocks.*.hook_resid_pre"