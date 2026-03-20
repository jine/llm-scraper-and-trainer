#!/usr/bin/env python3
"""Convert scraper JSONL output to training format for QLoRA fine-tuning."""

import argparse
import json
import random
from pathlib import Path


def load_scraper_data(input_dir: str) -> list[dict]:
    """Load and merge JSONL files from a directory."""
    all_entries = []
    seen_urls = set()

    input_path = Path(input_dir)
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.jsonl"))

    if not files:
        print(f"No .jsonl files found in {input_dir}")
        return []

    print(f"Found {len(files)} .jsonl files")

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                # Deduplicate by URL
                url = entry.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                # Skip entries with very short text
                text = entry.get("text", "").strip()
                if len(text) < 100:
                    continue

                all_entries.append(entry)

    return all_entries


def format_training_entry(entry: dict) -> dict:
    """Convert a scraper entry to training format.

    Format: Category as prompt, text as completion.
    The model learns to generate text conditioned on category.
    """
    categories = entry.get("categories", [])
    if categories:
        category_str = ", ".join(categories)
    else:
        category_str = "Allmänt"

    prompt = f"Kategori: {category_str}\nTitel: {entry.get('title', '')}\n\nSkriv text:"
    completion = f" {entry['text']}"

    return {
        "prompt": prompt,
        "completion": completion,
    }


def split_data(
    entries: list[dict], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split data into train and validation sets."""
    random.seed(seed)
    shuffled = entries.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def main():
    parser = argparse.ArgumentParser(
        description="Prepare scraper data for LLM training"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing .jsonl files, or a single .jsonl file",
    )
    parser.add_argument(
        "--output",
        default="trainer/data",
        help="Output directory (default: trainer/data)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting (default: 42)"
    )
    args = parser.parse_args()

    # Load data
    entries = load_scraper_data(args.input)
    print(f"Loaded {len(entries)} unique entries")

    if len(entries) < 10:
        print(
            "WARNING: Very few entries. Consider scraping more data for better results."
        )

    # Format for training
    formatted = [format_training_entry(e) for e in entries]

    # Split
    train, val = split_data(formatted, args.val_ratio, args.seed)
    print(f"Split: {len(train)} train, {len(val)} validation")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for entry in train:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for entry in val:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved to {train_path} and {val_path}")

    # Show sample
    sample = train if train else val
    if sample:
        print("\nSample entry:")
        print(json.dumps(sample[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
