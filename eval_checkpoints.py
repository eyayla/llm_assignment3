"""
Evaluation script for all 3 checkpoints:
  - Checkpoint 0: Base model (no fine-tuning)
  - Checkpoint 1: After Stage 1 (Alpaca fine-tuning)
  - Checkpoint 2: After Stage 2 (JSON instruction fine-tuning)

Generates responses for:
  - 100 Alpaca held-out prompts
  - 100 JSON held-out prompts (from json_instruct_dataset.jsonl)

Saves all outputs to eval_outputs/ directory.
"""

import os
import json
import torch
import dotenv
import random
import data_utils as du
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

dotenv.load_dotenv()

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.2-3B"

CHECKPOINTS = {
    "checkpoint_0_base": None,  # No adapter, just base model
    "checkpoint_1_alpaca": "./sft-lora-Llama-3.2-3B-alpaca-r32-a64-d0.05-lr2.0e-05-wd0.01/final",
    "checkpoint_2_json": "./stage2-lora-Llama-3.2-3B-json-r32-a64-lr2.0e-05/final",
}

JSON_DATASET_PATH = "./json_instruct_dataset.jsonl"
OUTPUT_DIR = "./eval_outputs"
N_ALPACA_EVAL = 100
N_JSON_EVAL = 50  # we only have 150 total, use 50 for eval
MAX_NEW_TOKENS = 256
SEED = 42

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def load_model_and_tokenizer(base_model, adapter_path=None):
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def format_alpaca_prompt(instruction, input_text=""):
    if input_text.strip():
        input_block = ", paired with an input that provides further context"
        input_section = f"### Input:\n{input_text}\n\n"
    else:
        input_block = ""
        input_section = ""
    return (
        f"Below is an instruction that describes a task{input_block}. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"{input_section}"
        f"### Response:\n"
    )


def load_alpaca_eval_set(n=100, seed=42):
    print("Loading Alpaca eval set...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.filter(lambda x: len(x["output"].strip()) > 0)
    random.seed(seed)
    indices = random.sample(range(len(dataset)), n)
    samples = [dataset[i] for i in indices]
    return samples


def load_json_eval_set(path, n=50, seed=42):
    print("Loading JSON eval set...")
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    random.seed(seed)
    random.shuffle(examples)
    return examples[:n]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    alpaca_eval = load_alpaca_eval_set(N_ALPACA_EVAL, SEED)
    json_eval   = load_json_eval_set(JSON_DATASET_PATH, N_JSON_EVAL, SEED)

    for ckpt_name, adapter_path in CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_name}")
        print(f"{'='*60}")

        model, tokenizer = load_model_and_tokenizer(BASE_MODEL, adapter_path)

        # --- Alpaca Eval ---
        alpaca_results = []
        for i, sample in enumerate(alpaca_eval):
            prompt = format_alpaca_prompt(sample["instruction"], sample.get("input", ""))
            response = generate_response(model, tokenizer, prompt)
            alpaca_results.append({
                "prompt_id": f"alpaca_{i:03d}",
                "instruction": sample["instruction"],
                "input": sample.get("input", ""),
                "reference": sample["output"],
                "checkpoint": ckpt_name,
                "response": response,
            })
            if (i + 1) % 10 == 0:
                print(f"  Alpaca: {i+1}/{N_ALPACA_EVAL}")

        alpaca_out = os.path.join(OUTPUT_DIR, f"{ckpt_name}_alpaca.json")
        with open(alpaca_out, "w") as f:
            json.dump(alpaca_results, f, indent=2)
        print(f"  Alpaca results saved to {alpaca_out}")

        # --- JSON Eval ---
        json_results = []
        for i, sample in enumerate(json_eval):
            instruction = sample["instruction"]
            input_text  = sample.get("input", "")
            prompt = format_alpaca_prompt(instruction, input_text)
            response = generate_response(model, tokenizer, prompt)

            # Check JSON validity
            is_valid = False
            try:
                json.loads(response.strip())
                is_valid = True
            except Exception:
                pass

            json_results.append({
                "prompt_id": f"json_{i:03d}",
                "task_type": sample.get("task_type", "unknown"),
                "instruction": instruction,
                "input": input_text,
                "reference": sample["output"],
                "checkpoint": ckpt_name,
                "response": response,
                "is_valid_json": is_valid,
            })
            if (i + 1) % 10 == 0:
                print(f"  JSON: {i+1}/{N_JSON_EVAL}")

        json_out = os.path.join(OUTPUT_DIR, f"{ckpt_name}_json.json")
        with open(json_out, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"  JSON results saved to {json_out}")

        # Print quick JSON validity summary
        valid_count = sum(1 for r in json_results if r["is_valid_json"])
        print(f"  JSON validity: {valid_count}/{N_JSON_EVAL} ({100*valid_count/N_JSON_EVAL:.1f}%)")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    print(f"\nAll evaluations complete. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
