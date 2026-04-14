#!/bin/bash
#SBATCH --job-name=sft-stage1
#SBATCH --partition=gpu4v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

conda activate /work/coa258/llm_env
cd /work/coa258/NLP/assignment3
mkdir -p logs
export CUDA_HOME=/usr/local/cuda-13.2

accelerate launch --num_processes 1 --mixed_precision bf16 train-sft.py --model_name_or_path meta-llama/Llama-3.2-3B --report_to none --fp16 False --bf16 True --eval_on_start False --max_steps 500 --save_steps 100 --eval_steps 100 --learning_rate 2e-5 --max_grad_norm 1.0
