# LLM Style Trainer

Fine-tune Llama-3.2-1B on Swedish text data using QLoRA to learn writing style from scraped content. Runs on consumer GPUs (8GB+ VRAM).

Model: [unsloth/Llama-3.2-1B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-bnb-4bit)

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

Download the base model locally (~1.5GB):

```bash
python download.py
```

This saves to `models/unsloth--Llama-3.2-1B-Instruct-bnb-4bit/`. You can then use the local path instead of downloading on each run.

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--model` | unsloth/Llama-3.2-1B-Instruct-bnb-4bit | HuggingFace model ID |
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
python train.py --base-model models/unsloth--Llama-3.2-1B-Instruct-bnb-4bit --data data/train.jsonl --output output/adapter
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--base-model` | unsloth/Llama-3.2-1B-Instruct-bnb-4bit | Base model name or path |
| `--epochs` | 1 | Training epochs |
| `--batch-size` | 1 | Per-device batch size |
| `--lr` | 2e-4 | Learning rate |
| `--max-length` | 512 | Max sequence length |
| `--lora-r` | 8 | LoRA rank |
| `--lora-alpha` | 16 | LoRA alpha |

### 3. Generate

#### Fine-tuned model (with adapter)

```bash
python generate.py --model output/adapter --category "Noveller" --title "En kort berättelse"
```

#### Base model (no adapter, for comparison)

```bash
python generate_base.py --category "Noveller" --title "En kort berättelse"
```

Options (both scripts):
| Flag | Default | Description |
|------|---------|-------------|
| `--model` | output/adapter (or base model for generate_base.py) | Model or adapter path |
| `--category` | (required) | Category to generate for |
| `--title` | | Optional title prompt |
| `--max-tokens` | 512 | Max tokens to generate |
| `--temperature` | 0.8 | Sampling temperature |
| `--top-p` | 0.9 | Top-p sampling |
| `--top-k` | 50 | Top-k sampling |
| `--repetition-penalty` | 1.2 | Repetition penalty (1.0 = disabled) |

## Notes

- Training on ~12k examples takes ~1h 45min on RTX 3060 Ti (~2 it/s)
- Adapter is small (~22MB) — easy to share or swap
- Llama-3.2-1B with 4-bit quantization fits in ~2-3GB VRAM for inference
- Uses Unsloth for 2x training speedup and efficient memory usage
- Inference uses transformers directly (Unsloth fast inference has compatibility issues with adapters)
