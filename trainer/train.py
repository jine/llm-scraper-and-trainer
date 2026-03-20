#!/usr/bin/env python3
"""QLoRA fine-tuning of Llama-3.1-8B with Unsloth for Swedish text data."""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_FUSED_CROSS_ENTROPY"] = "0"

import argparse
from pathlib import Path

from dotenv import load_dotenv
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

load_dotenv()


def load_dataset(path: str) -> Dataset:
    """Load JSONL training data."""
    import json

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)


def format_chat(example: dict) -> str:
    """Format prompt+completion into Llama chat template."""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['prompt']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['completion'].strip()}<|eot_id|>"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama-3.1-8B with Unsloth QLoRA"
    )
    parser.add_argument(
        "--base-model",
        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        help="Base model name or path (default: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)",
    )
    parser.add_argument(
        "--data", default="trainer/data/train.jsonl", help="Training data JSONL"
    )
    parser.add_argument(
        "--val-data", default="trainer/data/val.jsonl", help="Validation data JSONL"
    )
    parser.add_argument(
        "--output",
        default="trainer/output/adapter",
        help="Output directory for adapter",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Per-device batch size (default: 1)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length (default: 512)",
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument(
        "--lora-alpha", type=int, default=16, help="LoRA alpha (default: 16)"
    )
    args = parser.parse_args()

    # Check CUDA availability
    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Training will be very slow on CPU.")
        print("Make sure you have an NVIDIA GPU and CUDA drivers installed.")
        return

    # Clear any leftover GPU memory from previous runs
    torch.cuda.empty_cache()

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )

    # Auto-detect local model
    model_name = args.base_model
    if "/" in model_name and not Path(model_name).exists():
        local_path = Path("models") / model_name.replace("/", "--")
        if local_path.exists():
            print(f"Found local model at {local_path}, using it instead of downloading")
            model_name = str(local_path)

    print(f"Loading base model: {model_name}")

    # Load model with Unsloth (4-bit quantized)
    print("STEP 1: Loading base model...")
    token = os.environ.get("HUGGINGFACE_TOKEN")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_length,
        load_in_4bit=True,
        dtype=None,
        token=token,
    )
    print(f"  Model device: {next(model.parameters()).device}")
    free, total = torch.cuda.mem_get_info(0)
    print(
        f"  After base model: Free={free / 1024**3:.2f}GB, Alloc={torch.cuda.memory_allocated(0) / 1024**3:.2f}GB"
    )

    print("STEP 2: Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    free, total = torch.cuda.mem_get_info(0)
    print(
        f"  After LoRA: Free={free / 1024**3:.2f}GB, Alloc={torch.cuda.memory_allocated(0) / 1024**3:.2f}GB"
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )
    print(f"Model device after LoRA: {next(model.parameters()).device}")

    # Load datasets
    print(f"Loading training data from {args.data}")
    train_dataset = load_dataset(args.data)
    print(f"Training examples: {len(train_dataset)}")

    val_dataset = None
    if Path(args.val_data).exists():
        val_dataset = load_dataset(args.val_data)
        print(f"Validation examples: {len(val_dataset)}")

    # Format for chat
    train_dataset = train_dataset.map(
        lambda x: {"text": format_chat(x)},
        remove_columns=train_dataset.column_names,
    )

    if val_dataset:
        val_dataset = val_dataset.map(
            lambda x: {"text": format_chat(x)},
            remove_columns=val_dataset.column_names,
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=3407,
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_length,
    )

    # Train
    print("\nStarting training...")
    torch.cuda.empty_cache()
    trainer.train()

    # Save adapter
    print(f"Saving adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("Training complete!")


if __name__ == "__main__":
    main()
