# Sequential Instruction Tuning of a Small LLM with Strong-Model Judge Evaluation

**LLM & Agentic Systems — Assignment 3**  
**Student:** eyayla  
**Course:** LLM & Agentic Systems (Graduate)

---

## Overview

This project implements a two-stage instruction-tuning pipeline for a small language model (Llama 3.2 3B), evaluates the model at three checkpoints, and analyzes catastrophic forgetting using an LLM-as-a-Judge pipeline.

**Core research question:** Does fine-tuning on JSON-structured data after Alpaca instruction tuning preserve or degrade general instruction-following ability?

---

## Pipeline Summary

```
Base Model (Llama 3.2 3B)
        ↓
  Stage 1: Alpaca SFT (LoRA)
        ↓
  Stage 2: JSON Instruct SFT (LoRA, continued from Stage 1)
        ↓
  Evaluation at 3 Checkpoints + Judge Pipeline
```

---

## Repository Structure

```
assignment3/
├── train-sft.py              # Stage 1 training (Alpaca)
├── train_stage2.py           # Stage 2 training (JSON instruct)
├── generate_json_dataset.py  # Teacher-model JSON dataset generation
├── eval_checkpoints.py       # Generate responses at all 3 checkpoints
├── judge_eval.py             # LLM-as-a-Judge pairwise evaluation
├── config.py                 # Hyperparameters and training config
├── data_utils.py             # Alpaca data loading and formatting
├── eval.py                   # Interactive inference (fine-tuned model)
├── eval-base.py              # Interactive inference (base model)
├── run_stage1.sh             # SLURM batch script for Stage 1
├── run_stage2.sh             # SLURM batch script for Stage 2
├── run_eval.sh               # SLURM batch script for evaluation
├── run_judge.sh              # SLURM batch script for judge evaluation
├── json_instruct_dataset.jsonl  # Teacher-generated JSON dataset
├── logs/                     # Training and eval logs
├── eval_outputs/             # Model responses at each checkpoint
└── judge_outputs/            # Judge scores and summaries
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/eyayla/llm_assignment3.git
cd llm_assignment3

# Create environment
conda create -n llm_env python=3.10
conda activate llm_env

# Install dependencies
pip install -r requirements.txt

# Set up HuggingFace token (for Llama access)
huggingface-cli login
```

---

## Reproduction Steps

### Stage 1: Alpaca Fine-Tuning
```bash
sbatch run_stage1.sh
```

### Stage 2: JSON Instruction Fine-Tuning
```bash
sbatch run_stage2.sh
```

### Evaluation
```bash
sbatch run_eval.sh      # Generate responses at all 3 checkpoints
sbatch run_judge.sh     # Run LLM-as-a-Judge evaluation
```

### Generate JSON Dataset (teacher model)
```bash
python generate_json_dataset.py
```

---

## Model & Training Configuration

| Parameter | Value |
|-----------|-------|
| Student model | Llama 3.2 3B |
| Fine-tuning method | LoRA |
| Precision | BF16 |
| Stage 1 max steps | 500 |
| Stage 2 max steps | 200 |
| Learning rate | 2e-5 |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Max sequence length | 1024 |
| Teacher model | Llama 3.1 8B Instruct (UTSA API) |
| Judge model | Llama 3.1 8B Instruct (UTSA API) |

---

## Results Summary

| Checkpoint | JSON Validity | Alpaca Judge Win Rate |
|------------|--------------|----------------------|
| Checkpoint 0 (Base) | 24% | TBD |
| Checkpoint 1 (Alpaca) | 100% | TBD |
| Checkpoint 2 (JSON) | TBD | TBD |

*Full results in REPORT.md*

---

## Requirements

See `requirements.txt` for full dependency list.

Key packages:
- `torch>=2.0`
- `transformers>=4.40`
- `trl==0.9.6`
- `peft>=0.10`
- `datasets>=2.0`
- `accelerate>=0.30`
- `wandb`
