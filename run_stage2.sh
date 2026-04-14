#!/bin/bash
#SBATCH --job-name=sft-stage2
#SBATCH --partition=gpu4v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err

export PATH=/work/coa258/llm_env/bin:$PATH
cd /work/coa258/NLP/assignment3
mkdir -p logs
export CUDA_HOME=/usr/local/cuda-13.2
export STAGE1_CHECKPOINT=./sft-lora-Llama-3.2-3B-alpaca-r32-a64-d0.05-lr2.0e-05-wd0.01/final
export JSON_DATASET=./json_instruct_dataset.jsonl

accelerate launch --num_processes 1 --mixed_precision bf16 train_stage2.py --model_name_or_path meta-llama/Llama-3.2-3B --report_to none --fp16 False --bf16 True --eval_on_start False --max_steps 200 --save_steps 50 --eval_steps 50 --learning_rate 2e-5 --max_grad_norm 1.0
