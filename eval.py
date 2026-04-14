import os
import torch
import dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import data_utils as du

dotenv.load_dotenv()

# ---------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------
BASE_MODEL   = "meta-llama/Llama-3.2-3B"
ADAPTER_PATH = "./PreSaved_sft-lora-Llama-3.2-3B-alpaca-r32-a64-d0.05-lr1.0e-04-wd0.01/checkpoint-650"  # path to your saved adapter

INSTRUCTION  = "What is the capital of France?"
INPUT        = ""  # leave empty if no additional context is needed

MAX_NEW_TOKENS = 256
TEMPERATURE    = 0.7
TOP_P          = 0.9
# ---------------------------------------------------------


def load_model(base_model: str, adapter_path: str):
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("Model ready.\n")
    return model, tokenizer


def generate(model, tokenizer, instruction: str, input_text: str = "") -> str:
    # Reuse data_utils to format the prompt exactly as it was seen during training
    sample = {"instruction": instruction, "input": input_text}
    prompt = du.alpaca_row_to_prompt_eval(sample)["text"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the input prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    model, tokenizer = load_model(BASE_MODEL, ADAPTER_PATH)

    print("=" * 60)
    print(f"Instruction : {INSTRUCTION}")
    if INPUT:
        print(f"Input       : {INPUT}")
    print("=" * 60)

    response = generate(model, tokenizer, INSTRUCTION, INPUT)

    print(f"Response:\n{response}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Interactive loop — keep prompting until the user quits
    # ---------------------------------------------------------
    print("\nEntering interactive mode. Type 'quit' to exit.\n")
    while True:
        instruction = input("Instruction: ").strip()
        if instruction.lower() in ("quit", "exit", "q"):
            break
        input_text = input("Input (leave blank if none): ").strip()

        response = generate(model, tokenizer, instruction, input_text)
        print(f"\nResponse:\n{response}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()