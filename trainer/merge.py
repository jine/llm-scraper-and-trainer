#!/usr/bin/env python3
"""Merge LoRA adapter with Llama-3.2-1B base model using Unsloth."""

import argparse
import os

from dotenv import load_dotenv
from unsloth import FastLanguageModel
import torch

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base-model",
        default="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        help="Base model name (default: unsloth/Llama-3.2-1B-Instruct-bnb-4bit)",
    )
    parser.add_argument(
        "--adapter",
        default="trainer/output/adapter",
        help="Path to LoRA adapter (default: trainer/output/adapter)",
    )
    parser.add_argument(
        "--output",
        default="trainer/output/merged_model",
        help="Output directory for merged model (default: trainer/output/merged_model)",
    )
    parser.add_argument(
        "--method",
        choices=["merged_4bit", "merged_16bit"],
        default="merged_4bit",
        help="Merge method: merged_4bit (smaller, ~1.5GB) or merged_16bit (quality, ~2.5GB)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires GPU.")
        return

    torch.cuda.empty_cache()

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )

    token = os.environ.get("HUGGINGFACE_TOKEN")
    model_name = args.base_model

    if "/" in model_name and not os.path.exists(model_name):
        local_path = os.path.join("models", model_name.replace("/", "--"))
        if os.path.exists(local_path):
            print(f"Found local model at {local_path}, using it")
            model_name = local_path

    print(f"\nLoading base model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,
        token=token,
    )

    free, total = torch.cuda.mem_get_info(0)
    print(f"After base model: Free={free / 1024**3:.2f}GB")

    print(f"\nLoading adapter: {args.adapter}")
    model = FastLanguageModel.from_pretrained(model, args.adapter)

    free, total = torch.cuda.mem_get_info(0)
    print(f"After adapter: Free={free / 1024**3:.2f}GB")

    print(f"\nMerging with method: {args.method}...")
    print(f"Output directory: {args.output}")

    torch.cuda.empty_cache()

    model.save_pretrained_merged(args.output, tokenizer, args.method)

    print(f"\n✓ Merge complete! Model saved to: {args.output}")
    print(f"  Method: {args.method}")

    import subprocess

    result = subprocess.run(["du", "-sh", args.output], capture_output=True, text=True)
    print(f"  Size: {result.stdout.strip()}")


if __name__ == "__main__":
    main()
