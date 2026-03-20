# LLM Scraper & Trainer

A tool for scraping web content and fine-tuning an LLM to generate text in that style.

## Overview

This project consists of two main components:

1. **Web Scraper** (`scraper/`) - A concurrent web scraper that crawls websites, extracts title/text/categories, and outputs JSONL files for LLM training.

2. **LLM Trainer** (`trainer/`) - Fine-tunes Llama-3.1-8B using QLoRA on scraped Swedish text data to generate new content in learned styles.

## Architecture

```
scraper/           →  JSONL data  →  trainer/
  scraper.py           (pages)         prepare_data.py
  Concurrent crawler   dataset.jsonl   train.py (QLoRA)
  JSONL output         Categories     generate.py
```

## Features

### Scraper
- Concurrent crawling with ThreadPoolExecutor
- Extracts title, main text, and categories
- Respects same-domain links only
- Outputs individual pages + consolidated dataset
- Keyboard interrupt-safe with progress saving

### Trainer
- QLoRA fine-tuning for memory efficiency (8GB+ VRAM)
- Category-conditioned text generation
- Swedish text focus
- Small, shareable LoRA adapters
- Uses Unsloth for 2x training speedup

## Tech Stack

- Python 3.10+, requests, beautifulsoup4, lxml (scraper)
- PyTorch, transformers, peft, bitsandbytes, unsloth (trainer)

## Quick Start

### 1. Scrape Data

```bash
cd scraper
pip install -r requirements.txt
python scraper.py https://example.com --max-pages 100
```

### 2. Prepare Data

```bash
cd trainer
pip install -r requirements.txt
python prepare_data.py --input ../scraper/output/www.example.com/pages/ --output data/
```

### 3. Train

```bash
python train.py --data data/train.jsonl --output output/adapter
```

### 4. Generate

```bash
python generate.py --adapter output/adapter --category "Category" --title "Optional Title"
```

## Scraper Options

| Flag | Default | Description |
|------|---------|-------------|
| `--workers` | 3 | Max concurrent requests |
| `--delay` | 1.5 | Seconds between requests per worker |
| `--max-pages` | 0 | Max pages to scrape (0 = unlimited) |
| `--pattern` | `^/\w+/\d+/\S+/?$` | Regex for URL paths to save |

## Trainer Options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 1 | Training epochs |
| `--batch-size` | 1 | Per-device batch size |
| `--lr` | 2e-4 | Learning rate |
| `--max-length` | 512 | Max sequence length |
| `--lora-r` | 8 | LoRA rank |

See [scraper/README.md](scraper/README.md) and [trainer/README.md](trainer/README.md) for detailed documentation.
