"""
Turn website_corpus.txt into the parquet shards nanochat trains on.

nanochat's pretraining dataloader reads parquet files with a 'text' column
from $NANOCHAT_BASE_DIR/base_data_climbmix/ (see nanochat/dataset.py).
The LAST file alphabetically is used as the validation split, the rest as train.

This writes two shards:
  shard_00000.parquet  train documents (90%)
  shard_00001.parquet  validation documents (10%, read last -> val split)

Usage (from the nanochat repo root, after `uv sync --extra cpu`):
  NANOCHAT_BASE_DIR=$HOME/.cache/nanocat python nanocat/make_parquet.py
"""
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq


def get_base_dir():
    # Same rule as nanochat.common.get_base_dir, but without importing torch:
    # NANOCHAT_BASE_DIR env var, else ~/.cache/nanochat.
    if os.environ.get("NANOCHAT_BASE_DIR"):
        d = os.environ["NANOCHAT_BASE_DIR"]
    else:
        d = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(here, "website_corpus.txt")
    with open(corpus_path, encoding="utf-8") as f:
        corpus = f.read()

    # One document per blank-line-separated block; drop stubs.
    docs = [d.strip() for d in corpus.split("\n\n") if len(d.strip()) > 200]
    rng = random.Random(42)
    rng.shuffle(docs)
    n_val = max(1, len(docs) // 10)
    val_docs, train_docs = docs[:n_val], docs[n_val:]
    print(f"documents: {len(docs)} (train {len(train_docs)}, val {len(val_docs)})")
    print(f"train chars: {sum(len(d) for d in train_docs):,}, "
          f"val chars: {sum(len(d) for d in val_docs):,}")

    data_dir = os.path.join(get_base_dir(), "base_data_climbmix")
    os.makedirs(data_dir, exist_ok=True)
    for name, chunk in [("shard_00000.parquet", train_docs),
                        ("shard_00001.parquet", val_docs)]:
        table = pa.table({"text": chunk})
        path = os.path.join(data_dir, name)
        pq.write_table(table, path)
        print(f"wrote {path} ({len(chunk)} docs)")
    print("Done. Tokenizer + pretraining will now read these shards.")


if __name__ == "__main__":
    main()
