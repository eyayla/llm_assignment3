import os
import json
import dotenv
import config
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

dotenv.load_dotenv()


ALPACA_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task{input_block}. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_section}"
    "### Response:\n"
)


def format_json_example(sample):
    instruction = sample["instruction"]
    input_text = sample.get("input", "").strip()
    output = sample["output"]

    if input_text:
        input_block = ", paired with an input that provides further context"
        input_section = f"### Input:\n{input_text}\n\n"
    else:
        input_block = ""
        input_section = ""

    prompt = ALPACA_PROMPT_TEMPLATE.format(
        input_block=input_block,
        instruction=instruction,
        input_section=input_section,
    )
    return {"text": prompt + output}


def load_json_dataset(path: str, validation_size: float = 0.1, seed: int = 42):
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(format_json_example, batched=False, remove_columns=dataset.column_names)

    split = dataset.train_test_split(test_size=validation_size, seed=seed)
    return split["train"], split["test"]


def main():
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

    model_args, data_args, lora_args, train_args = config.get_config_classes(training_type="sft")

    # Load JSON dataset
    json_dataset_path = os.environ.get("JSON_DATASET", "json_instruct_dataset.jsonl")
    train_set, val_set = load_json_dataset(json_dataset_path, validation_size=0.1)

    print(f"JSON Train size: {len(train_set)}")
    print(f"JSON Val size:   {len(val_set)}")

    # Stage 1 checkpoint to continue from
    stage1_checkpoint = os.environ.get("STAGE1_CHECKPOINT", None)

    output_dir = (
        f"./stage2-lora-{model_args.model_name_or_path.split('/')[-1]}"
        f"-json"
        f"-r{lora_args.r}"
        f"-a{lora_args.lora_alpha}"
        f"-lr{train_args.learning_rate:.1e}"
    )
    train_args.output_dir = output_dir
    train_args.run_name = output_dir
    print("Output directory:", output_dir)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )

    # Load Stage 1 LoRA adapter if provided, else train from scratch
    if stage1_checkpoint and os.path.exists(stage1_checkpoint):
        print(f"Loading Stage 1 adapter from: {stage1_checkpoint}")
        model = PeftModel.from_pretrained(model, stage1_checkpoint, is_trainable=True)
    else:
        print("No Stage 1 checkpoint found, applying fresh LoRA...")
        peft_config = LoraConfig(
            r=lora_args.r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.target_modules,
            lora_dropout=lora_args.lora_dropout,
            task_type=lora_args.task_type,
        )
        model = get_peft_model(model, peft_config)

    if is_main_process:
        model.print_trainable_parameters()

    peft_config = LoraConfig(
        r=lora_args.r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.target_modules,
        lora_dropout=lora_args.lora_dropout,
        task_type=lora_args.task_type,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(os.path.join(train_args.output_dir, "final"))
    print("Stage 2 training complete. Adapter saved to:", output_dir)


if __name__ == "__main__":
    main()
