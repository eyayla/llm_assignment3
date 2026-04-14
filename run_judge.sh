#!/bin/bash
#SBATCH --job-name=judge-eval
#SBATCH --partition=gpu4v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/judge_%j.out
#SBATCH --error=logs/judge_%j.err

export PATH=/work/coa258/llm_env/bin:$PATH
cd /work/coa258/NLP/assignment3
mkdir -p logs judge_outputs

python judge_eval.py
