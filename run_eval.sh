#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu4v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

export PATH=/work/coa258/llm_env/bin:$PATH
cd /work/coa258/NLP/assignment3
mkdir -p logs eval_outputs

export CUDA_HOME=/usr/local/cuda-13.2

python eval_checkpoints.py
