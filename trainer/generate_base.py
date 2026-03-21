#!/usr/bin/env python3
"""Generate text using base Llama-3.2-1B model (no adapter)."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with base Llama model (no adapter)"
    )
    parser.add_argument(
        "--model",
        default="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        help="Model name (default: unsloth/Llama-3.2-1B-Instruct-bnb-4bit)",
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
        "--temperature", type=float, default=0.8, help="Temperature (default: 0.8)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p sampling (default: 0.9)"
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Top-k sampling (default: 50)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty (default: 1.2, 1.0 = disabled)",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

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
            max_length=None,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][input_ids.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    print("\nGenerated text:")
    print("-" * 60)
    print(text)
    print("-" * 60)


if __name__ == "__main__":
    main()
