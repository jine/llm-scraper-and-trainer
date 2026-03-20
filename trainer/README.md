# LLM Style Trainer

Fine-tune Llama-3.1-8B on Swedish text data using QLoRA to learn writing style from scraped content. Runs on consumer GPUs (8GB+ VRAM).

Model: [unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit)

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable NVIDIA GPU with 8+ GB VRAM.

## Pipeline

### 0. Download Model

Download the base model locally (~4.5GB):

```bash
python download.py
```

This saves to `models/unsloth--Meta-Llama-3.1-8B-Instruct-bnb-4bit/`. You can then use the local path instead of downloading on each run.

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--model` | unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit | HuggingFace model ID |
| `--output` | models | Download directory |

### 1. Prepare Data

Convert scraper output to training format:

```bash
python prepare_data.py --input scraper/output/www.site.tld/pages/ --output data/
```

Accepts a directory (globs all `.jsonl` files) or a single file.

### 2. Train

Fine-tune with QLoRA via Unsloth:

```bash
# From HuggingFace (downloads on first run)
python train.py --data data/train.jsonl --output output/adapter

# From local download
python train.py --base-model models/unsloth--Meta-Llama-3.1-8B-Instruct-bnb-4bit --data data/train.jsonl --output output/adapter
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--base-model` | unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit | Base model name or path |
| `--epochs` | 1 | Training epochs |
| `--batch-size` | 1 | Per-device batch size |
| `--lr` | 2e-4 | Learning rate |
| `--max-length` | 512 | Max sequence length |
| `--lora-r` | 8 | LoRA rank |
| `--lora-alpha` | 16 | LoRA alpha |

### 3. Generate

Generate text using the fine-tuned model:

```bash
python generate.py --adapter output/adapter --category "Noveller" --title "En kort berättelse"
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--base-model` | unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit | Base model name or path |
| `--category` | (required) | Category to generate for |
| `--title` | | Optional title prompt |
| `--max-tokens` | 512 | Max tokens to generate |
| `--temperature` | 0.8 | Sampling temperature |
| `--top-p` | 0.9 | Top-p sampling |

## Notes

- Training on ~24k examples takes ~15-20 hours on RTX 3060 Ti
- Adapter is small (~20MB) — easy to share or swap
- Llama-3.1-8B with 4-bit quantization fits in ~5.4GB VRAM
- Uses Unsloth for 2x training speedup and efficient memory usage
