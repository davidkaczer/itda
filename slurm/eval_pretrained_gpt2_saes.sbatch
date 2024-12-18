#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos short
#SBATCH -t 00-12:00
#SBATCH --gres=gpu
#SBATCH -o logs/eval_pretrained_gpt2_saes-%A.out
#SBATCH --mem 8G

module load cuda/11.3
source /path/to/your/venv/bin/activate

python benchmark_sae.py \
  --model gpt2 \
  --pretrained_saes \
  --eval_types core scr tpp sparse_probing
