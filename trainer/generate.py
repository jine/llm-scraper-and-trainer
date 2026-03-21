#!/usr/bin/env python3
"""Generate text using fine-tuned Llama-3.2-1B model via Unsloth."""

import argparse
import os

from dotenv import load_dotenv
from unsloth import FastLanguageModel
import torch
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
        help="Path to merged model OR adapter (use --mode to specify)",
    )
    parser.add_argument(
        "--mode",
        choices=["merged", "adapter"],
        default="merged",
        help="Load as merged model or LoRA adapter (default: merged)",
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
    args = parser.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN")

    if args.mode == "merged":
        if not args.model:
            args.model = "trainer/output/merged_model"
        print(f"Loading merged model: {args.model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=1024,
            load_in_4bit=True,
            dtype=None,
            token=token,
        )
    else:
        model_name = args.base_model
        if not args.model:
            args.model = "trainer/output/adapter"
        print(f"Loading base model: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            load_in_4bit=True,
            dtype=None,
            token=token,
        )
        print(f"Loading adapter: {args.model}")
        model = PeftModel.from_pretrained(model, args.model)

    FastLanguageModel.for_inference(model)

    prompt = f"Kategori: {args.category}"
    if args.title:
        prompt += f"\nTitel: {args.title}"
    prompt += "\n\nSkriv text:"

    print(f"\nPrompt: {prompt}\n")

    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)

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

    generated = outputs[0][input_ids.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    print("Generated text:")
    print("-" * 60)
    print(text)
    print("-" * 60)


if __name__ == "__main__":
    main()
