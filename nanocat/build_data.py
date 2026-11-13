"""
Build NanoCat training data from Sushma's website.

Reads:
  - <site-dir>/index.html                  (homepage: 16 tabs, quotes, notes)
  - <site-dir>/index_conv_cat.html         (Cat page: the 112 FAQs, embedded as FAQS JSON)
  - <site-dir>/srt-files/*.srt            (video narration, in Sushma's own voice)

Writes (into <out-dir>, default: this folder):
  - website_corpus.txt          plain text for pretraining (parquet step reads this)
  - nanocat_faq_train.jsonl     100 Q&A conversations for SFT (CustomJSON format)
  - nanocat_faq_val.jsonl       12 Q&A conversations held out for SFT validation

Usage (from the nanochat repo root):
  python nanocat/build_data.py --site-dir /path/to/website/files --out-dir nanocat

The generated .txt/.jsonl files are committed to the nanocat branch, so
re-running this is only needed when the website changes.
"""
import argparse
import glob
import html
import json
import os
import random
import re
import sys


def strip_html(text):
    """Remove tags, unescape entities, collapse whitespace."""
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ([,.!?;:])", r"\1", text)  # no space before punctuation
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def homepage_text(site_dir):
    path = os.path.join(site_dir, "index.html")
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    # Keep only the visible page body: drop head, scripts and styles first.
    body = re.search(r"(?is)<body[^>]*>(.*)</body>", raw)
    text = strip_html(body.group(1) if body else raw)
    # NanoCat must not learn to talk about her court case: drop any sentence
    # with legal/dismissal wording (the resilience framing itself stays).
    legal = re.compile(r"(?i)plaintiff|litigat|dismissal|stipulation|eeoc|\bada\b")
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not legal.search(sentence):
            kept.append(sentence)
    return " ".join(kept)


def faqs(site_dir):
    """The 112 FAQs, parsed from the FAQS JSON embedded in index_conv_cat.html."""
    path = os.path.join(site_dir, "index_conv_cat.html")
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    m = re.search(r"const FAQS = (\[.*?\]);", raw, re.DOTALL)
    if not m:
        raise SystemExit("FAQS JSON not found in index_conv_cat.html")
    entries = json.loads(m.group(1))
    out = []
    for e in entries:
        q = strip_html(e["q"]).replace("\n", " ").strip()
        a = strip_html(e["a"]).replace("\n", " ").strip()
        a = re.sub(r" +", " ", a)
        if q and a:
            out.append({"topic": e.get("t", ""), "q": q, "a": a})
    return out


def narration_text(site_dir):
    """All video narration, from the .srt sidecars (timestamps/numbers stripped)."""
    parts = []
    for path in sorted(glob.glob(os.path.join(site_dir, "srt-files", "*.srt"))):
        with open(path, encoding="utf-8") as f:
            lines = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.isdigit():
                    continue
                if "-->" in line:
                    continue
                # strip inline SRT tags like <i>...</i> and {an8}
                line = re.sub(r"<[^>]+>", "", line)
                line = re.sub(r"\{[^}]*\}", "", line)
                lines.append(line)
        if lines:
            parts.append(" ".join(lines))
    return parts


def main():
    ap = argparse.ArgumentParser(description="Build NanoCat training data from the website")
    ap.add_argument("--site-dir", default=os.path.expanduser("~/workspace/homepage-videos"),
                    help="folder with index.html, index_conv_cat.html, srt-files/")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="where to write website_corpus.txt and the FAQ jsonl files")
    args = ap.parse_args()

    home = homepage_text(args.site_dir)
    faq_list = faqs(args.site_dir)
    narrations = narration_text(args.site_dir)
    print(f"homepage chars: {len(home):,}")
    print(f"FAQs: {len(faq_list)}")
    print(f"narration files: {len(narrations)}, chars: {sum(len(n) for n in narrations):,}")

    # ---- pretraining corpus: homepage + narration + FAQs as prose ----
    docs = []
    docs.append("Sushma Akoju's research website. " + home)
    for n in narrations:
        docs.append(n)
    for f in faq_list:
        docs.append(f"Question: {f['q']}\nAnswer: {f['a']}")
    corpus = "\n\n".join(docs)
    # No trailing whitespace on any line: git apply/diff --check complain
    # about it, and the homepage text leaves trailing spaces behind.
    corpus = "\n".join(line.rstrip() for line in corpus.split("\n"))
    corpus_path = os.path.join(args.out_dir, "website_corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"wrote {corpus_path} ({len(corpus):,} chars, ~{len(corpus.split()):,} words)")

    # ---- SFT conversations: one user/assistant turn per FAQ ----
    rng = random.Random(42)
    idx = list(range(len(faq_list)))
    rng.shuffle(idx)
    val_idx = set(idx[:12])
    train_path = os.path.join(args.out_dir, "nanocat_faq_train.jsonl")
    val_path = os.path.join(args.out_dir, "nanocat_faq_val.jsonl")
    with open(train_path, "w", encoding="utf-8") as ft, open(val_path, "w", encoding="utf-8") as fv:
        for i, f in enumerate(faq_list):
            line = json.dumps([
                {"role": "user", "content": f["q"]},
                {"role": "assistant", "content": f["a"]},
            ], ensure_ascii=False)
            (fv if i in val_idx else ft).write(line + "\n")
    print(f"wrote {train_path} (100) and {val_path} (12)")


if __name__ == "__main__":
    main()
