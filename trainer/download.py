#!/usr/bin/env python3
"""Download Llama-3.1-8B-Instruct model locally."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Download Llama-3.1-8B-Instruct model")
    parser.add_argument(
        "--model",
        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        help="Model name on HuggingFace (default: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)",
    )
    parser.add_argument(
        "--output",
        default="models",
        help="Output directory (default: models)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision/branch (default: main)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    model_name = args.model.replace("/", "--")
    local_dir = output_dir / model_name

    print(f"Downloading {args.model} to {local_dir}")
    print("This may take a while (~4.5GB)...\n")

    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        print("Using HuggingFace token from .env")

    snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
    )

    print(f"\nModel saved to {local_dir}")
    print(f"\nTo use with train.py:")
    print(f"  python train.py --base-model {local_dir}")
    print(f"\nTo use with generate.py:")
    print(
        f"  python generate.py --base-model {local_dir} --adapter output/adapter --category Noveller"
    )


if __name__ == "__main__":
    main()
