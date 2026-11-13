# NanoCat

Branch `nanocat`: train a tiny **test** model on Sushma's website text and
teach it to answer as Cat — CPU-only, on a Windows laptop (WSL).

- `website_corpus.txt` — pretraining text (~25k words: homepage tabs,
  video narration, Cat FAQs). Built by `build_data.py` from the website.
- `nanocat_faq_train.jsonl` / `nanocat_faq_val.jsonl` — the 112 Cat FAQs
  as user/assistant conversations (100 train / 12 val) for SFT.
- `build_data.py` — rebuilds the corpus + FAQ files from the website
  (`--site-dir` points at the folder with `index.html`,
  `index_conv_cat.html`, `srt-files/`). Drops legal/dismissal sentences.
- `make_parquet.py` — turns the corpus into the parquet shards nanochat
  trains on (`$NANOCHAT_BASE_DIR/base_data_climbmix/`).
- `WINDOWS_CPU.md` — **start here**: which data changes, setup, the
  one-command run, what to expect, troubleshooting.
- `runs/nanocat_cpu.sh` — the full CPU pipeline in one command.

The only code change vs stock nanochat is the SFT mixture in
`scripts/chat_sft.py` (marked "NanoCat"): SmolTalk/MMLU/GSM8K/spelling
replaced by the Cat FAQs, so SFT downloads nothing big.
