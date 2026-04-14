import os
import dotenv

import config
import data_utils as du

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


# ---------------------------------------------------------
# 0. Environment
# ---------------------------------------------------------

dotenv.load_dotenv()


# ---------------------------------------------------------
#  Training
# ---------------------------------------------------------

def main():
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

    # ---------------------------------------------------------
    # 1. Config
    # ---------------------------------------------------------
    model_args, data_args, lora_args, train_args = config.get_config_classes(training_type="sft")

    # ---------------------------------------------------------
    # 2. Load the dataset
    # ---------------------------------------------------------
    train_set, val_set = du.prepare_alpaca_data(
        dataset_name=data_args.dataset_name,
        process_func=du.alpaca_row_to_text_train,
        validation_size=data_args.validation_size,
        seed=data_args.dataset_ops_seed,
    )

    output_dir = (
        f"./sft-lora-{model_args.model_name_or_path.split('/')[-1]}"
        f"-alpaca"
        f"-r{lora_args.r}"
        f"-a{lora_args.lora_alpha}"
        f"-d{lora_args.lora_dropout:.2f}"
        f"-lr{train_args.learning_rate:.1e}"
        f"-wd{train_args.weight_decay:.2f}"
    )
    train_args.output_dir = output_dir
    train_args.run_name = output_dir
    print("Output directory:", train_args.output_dir)

    # ---------------------------------------------------------
    # 3. Load tokenizer & model
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token  # Llama-3.2 has no pad token by default

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # device_map="auto",  # Let accelerate handle multi-GPU
        torch_dtype=torch.bfloat16,
    )

    print("Max sequence length:", model.config.max_position_embeddings)
    print(f"Train dataset size:      {len(train_set)}")
    print(f"Validation dataset size: {len(val_set)}")

    # ---------------------------------------------------------
    # 4. LoRA
    # ---------------------------------------------------------
    peft_config = LoraConfig(
        r=lora_args.r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.target_modules,
        lora_dropout=lora_args.lora_dropout,
        task_type=lora_args.task_type,  # CAUSAL_LM
    )

    model = get_peft_model(model, peft_config)

    if is_main_process:
        model.print_trainable_parameters()

    # ---------------------------------------------------------
    # 5. Train
    # ---------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    trainer.save_model(os.path.join(train_args.output_dir, "final"))  # Saves LoRA adapter only (not the base model)
    print("Training complete. LoRA adapter saved to:", train_args.output_dir)


if __name__ == "__main__":
    main()
