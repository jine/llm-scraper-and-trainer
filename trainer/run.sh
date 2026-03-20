#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# 1. Activate venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# 3. Check if model is downloaded
MODEL_DIR="models/unsloth--Llama-3.2-1B-Instruct-bnb-4bit"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Model not found locally, downloading..."
    python download.py
else
    echo "Model already downloaded at $MODEL_DIR"
fi

# 4. Prepare data
echo "Preparing training data..."
python prepare_data.py --input scraper/output/www.site.tld/pages/ --output data/

# 5. Wait for user to confirm
echo ""
echo "Ready to train. Press Enter to start..."
read -r

python train.py --data data/train.jsonl --output output/adapter --base-model "$MODEL_DIR" --max-length 512
