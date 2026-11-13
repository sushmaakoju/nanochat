# NanoCat — train a test model on Sushma's website text, CPU-only (Windows laptop)

Branch `nanocat` of `sushmaakoju/nanochat`. A tiny depth-4 test model that
learns Sushma's website text and answers as Cat. CPU-only, runs in WSL.

## What changes vs stock nanochat (which data to change)

| What | Stock nanochat | NanoCat branch |
|---|---|---|
| Pretraining data | ClimbMix-400B, downloaded from HuggingFace (`python -m nanochat.dataset`) | Your website text: `nanocat/website_corpus.txt` (~25k words: homepage tabs + video narration + Cat FAQs). Turned into parquet by `nanocat/make_parquet.py`. **Do not run `python -m nanochat.dataset`** — it would download ClimbMix into the same folder. |
| Tokenizer | vocab 32768, trained on 2B chars | vocab **4096**, trained on the website corpus (`--vocab-size 4096 --max-chars 200000`) |
| Base model | depth 24, 8xH100 | **depth 4**, `--max-seq-len=512`, `--device-batch-size=1`, `--total-batch-size=512` |
| SFT data | SmolTalk + MMLU + GSM8K + spelling (large downloads) | **Only your 112 Cat FAQs** (`nanocat/nanocat_faq_train.jsonl`, 100 rows x3 epochs; 12 held out in `nanocat_faq_val.jsonl`). See the one code change in `scripts/chat_sft.py` (mixture replaced, clearly marked "NanoCat"). |
| Install | `uv sync --extra gpu` | **`uv sync --extra cpu`** (installs CPU torch `2.9.1+cpu`; verified it resolves) |
| Where files go | `~/.cache/nanochat` | **`~/.cache/nanocat`** (set `NANOCHAT_BASE_DIR`). Keeps NanoCat apart from any stock run. |

Data hygiene: the corpus builder (`nanocat/build_data.py`) drops any sentence
with legal/dismissal wording (plaintiff / litigated / dismissal / stipulation),
so the model never learns to talk about the court case. The resilience
framing itself stays in. No "daughter of Holly Golightly" line anywhere —
it was retracted and is not in the source pages.

## Setup (Windows, WSL)

You already use WSL — everything below runs in your Ubuntu WSL terminal,
not in PowerShell or cmd.

```bash
# 1) get the repo and the branch (do this once)
git clone https://github.com/sushmaakoju/nanochat.git
cd nanochat
git checkout -b nanocat
# (the nanocat files come with this branch; if you are applying them by hand,
#  copy nanocat/, runs/nanocat_cpu.sh and scripts/chat_sft.py into the repo)

# 2) push the branch to your GitHub (so it is saved)
git push -u origin nanocat
```

## Run it (one command)

```bash
bash runs/nanocat_cpu.sh
```

That does, in order: CPU venv (`uv sync --extra cpu`) → parquet shards from
`website_corpus.txt` → tokenizer (vocab 4096) → 20-step smoke-test pretrain
→ 300-step test pretrain → SFT on the FAQs → a test chat (`Who are you?`)
→ interactive chat. Leave the terminal open; it takes roughly **30–60
minutes** on a laptop CPU, most of it `torch.compile` (first minutes) plus
training. Run it in `screen` or `tmux` if you step away.

Step by step (same thing, if you prefer to run each piece yourself):

```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanocat"
uv sync --extra cpu && source .venv/bin/activate
python nanocat/make_parquet.py
cp nanocat/nanocat_faq_train.jsonl nanocat/nanocat_faq_val.jsonl "$NANOCHAT_BASE_DIR/"
python -m scripts.tok_train --vocab-size 4096 --max-chars 200000
python -m scripts.tok_eval
python -m scripts.base_train --device-type cpu --depth=4 --max-seq-len=512 \
  --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 \
  --total-batch-size=512 --num-iterations=20 --sample-every=-1 --save-every=-1
python -m scripts.base_train --device-type cpu --depth=4 --max-seq-len=512 \
  --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 \
  --total-batch-size=512 --num-iterations=300 --sample-every=100
python -m scripts.chat_sft --device-type cpu --num-iterations=80 \
  --max-seq-len=512 --device-batch-size=1 --total-batch-size=512 \
  --eval-tokens=512 --eval-every=-1 --chatcore-every=-1
python -m scripts.chat_cli --device-type cpu -p "Who are you?"
python -m scripts.chat_cli --device-type cpu   # interactive chat
```

Notes on the flags: `--device-type cpu` is explicit (autodetect would also
pick CPU on your laptop, since there is no CUDA). `--core-metric-every=-1`
and `--chatcore-every=-1` skip the heavy benchmark evals. `--eval-tokens=512`
keeps validation tiny. No `--run` is passed, so wandb logging is off
(`dummy`); pass `--run=nanocat-d4` only after `wandb login`.

## Talk to the test model

```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanocat"
source .venv/bin/activate
python -m scripts.chat_cli --device-type cpu -p "Who are you?"
python -m scripts.chat_cli --device-type cpu -p "What is the bat example?"
python -m scripts.chat_cli --device-type cpu        # interactive
# pretty web UI instead (then open the printed http://localhost:8000 URL):
python -m scripts.chat_web --device-type cpu
```

Try the FAQ questions first — those are the ones it was trained to answer.

## What to honestly expect

This is a **test model**, not a real chatbot. Depth 4, ~25k words of text,
300 pretraining steps + 80 SFT steps. It will:

- parrot Cat phrases from the FAQs ("I'm Cat — the reincarnation of the
  cat from Breakfast at Tiffany's..."), sometimes word-for-word;
- answer the 112 FAQ questions approximately, and ramble or repeat on
  anything else;
- show falling training loss, which is the real signal that the pipeline
  works end to end.

It will **not** reason, generalise, or hold a real conversation like the
Cat page (which answers from a fixed list). If you want a genuinely good
NanoCat, the next step is a GPU run (even one cheap cloud GPU) with a
bigger model and more text — the branch is built so the same data and
commands scale up by changing `--depth` and `--num-iterations`.

## If something goes wrong

- **`torch.compile` very slow or failing on CPU**: run with compiling off:
  `TORCH_COMPILE_DISABLE=1 python -m scripts.base_train ...` (same for
  `chat_sft`). Training still works, just slower per step.
- **Line endings / `^M` errors**: if `bash runs/nanocat_cpu.sh` fails with
  a "bad interpreter" or `^M` error, the script got Windows (CRLF) line
  endings — this happens if the patch was applied outside WSL. Fix it with
  `sed -i 's/\r$//' runs/nanocat_cpu.sh` and re-run. Easiest is to do the
  `git clone` / `git apply` inside WSL in the first place. (In WSL, your
  Windows Downloads folder is `/mnt/c/Users/<you>/Downloads/`.)
- **Out of memory**: you are already at the minimum (`--device-batch-size=1`,
  `--max-seq-len=512`). Close other apps and re-run; checkpoints are saved
  at the end of each run, so a crash only loses the current run.
- **It still downloads something big**: you ran `python -m nanochat.dataset`
  (don't) or the stock `runs/speedrun.sh` (don't — that is the GPU script).
  Only `runs/nanocat_cpu.sh` on this branch.
- **`uv sync --extra gpu` by mistake**: that installs the CUDA torch, which
  won't run on your laptop. Re-run `uv sync --extra cpu`.
- **Resume**: checkpoints live in `$NANOCHAT_BASE_DIR/base_checkpoints/d4/`
  and `$NANOCHAT_BASE_DIR/chatsft_checkpoints/d4/`. Re-running a command
  starts fresh unless you add `--resume-from-step <N>`.
- **Regenerating the data** (after the website changes):
  `python nanocat/build_data.py --site-dir /path/to/website/files`
  then `python nanocat/make_parquet.py` again.
