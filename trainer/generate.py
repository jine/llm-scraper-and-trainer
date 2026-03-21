#!/usr/bin/env python3
"""Generate text using fine-tuned Llama-3.2-1B model via transformers."""

import argparse
import os

from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with fine-tuned Llama model"
    )
    parser.add_argument(
        "--base-model",
        default="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        help="Base model name (default: unsloth/Llama-3.2-1B-Instruct-bnb-4bit)",
    )
    parser.add_argument(
        "--model",
        default="output/adapter",
        help="Path to LoRA adapter (default: output/adapter)",
    )
    parser.add_argument("--category", required=True, help="Category to generate for")
    parser.add_argument("--title", default="", help="Optional title for the prompt")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p sampling (default: 0.9)"
    )
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading adapter: {args.model}")
    model = PeftModel.from_pretrained(model, args.model)

    prompt = f"Kategori: {args.category}"
    if args.title:
        prompt += f"\nTitel: {args.title}"
    prompt += "\n\nSkriv text:"

    print(f"\nPrompt: {prompt}\n")

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    generated = outputs[0][input_ids.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    print("\nGenerated text:")
    print("-" * 60)
    print(text)
    print("-" * 60)


if __name__ == "__main__":
    main()
