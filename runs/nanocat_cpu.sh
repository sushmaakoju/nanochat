#!/bin/bash
# NanoCat: train a tiny TEST model on Sushma's website text, CPU-only.
# Run from the nanochat repo root, inside WSL (Ubuntu) on a Windows laptop.
#   bash runs/nanocat_cpu.sh
# Takes roughly 30-60 minutes on a laptop CPU (mostly torch.compile + training).
# Everything lands in $NANOCHAT_BASE_DIR (default ~/.cache/nanocat), so it
# never touches a stock nanochat run. See nanocat/WINDOWS_CPU.md for the
# step-by-step and what to expect.
set -e
cd "$(dirname "$0")/.."

export NANOCHAT_BASE_DIR="$HOME/.cache/nanocat"
mkdir -p "$NANOCHAT_BASE_DIR"

# --- 1) venv with CPU torch (NOT the gpu extra) ---
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate

# --- 2) data: website text -> parquet shards (train + val) ---
python nanocat/make_parquet.py
# SFT conversations go in the base dir, where chat_sft.py (nanocat branch) reads them
cp nanocat/nanocat_faq_train.jsonl nanocat/nanocat_faq_val.jsonl "$NANOCHAT_BASE_DIR/"

# --- 3) tokenizer: small vocab for a small corpus ---
python -m scripts.tok_train --vocab-size 4096 --max-chars 200000
python -m scripts.tok_eval

# --- 4) pretraining: depth-4 test model, CPU ---
# 4a) 20-step smoke test first: proves the pipeline works before the longer run.
python -m scripts.base_train --device-type cpu --depth=4 --max-seq-len=512 \
  --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 \
  --total-batch-size=512 --num-iterations=20 --sample-every=-1 --save-every=-1
# 4b) the real test model: 300 steps (~3 epochs over the ~25k-word corpus).
python -m scripts.base_train --device-type cpu --depth=4 --max-seq-len=512 \
  --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 \
  --total-batch-size=512 --num-iterations=300 --sample-every=100

# --- 5) SFT: teach it to answer as Cat, from the 112 FAQs ---
python -m scripts.chat_sft --device-type cpu --num-iterations=80 \
  --max-seq-len=512 --device-batch-size=1 --total-batch-size=512 \
  --eval-tokens=512 --eval-every=-1 --chatcore-every=-1

# --- 6) talk to it ---
python -m scripts.chat_cli --device-type cpu -p "Who are you?"
echo "---"
echo "Now try it interactively:"
python -m scripts.chat_cli --device-type cpu
