"""Collect and clean training text (dialogue, science, history) into train.txt.

Setup:   pip install datasets
Run:     python collect.py
Option:  --force   overwrite an existing train.txt

Your current train.txt is already finished, so this refuses to run while it
exists. If you build a new train.txt, delete tok.json and model_2.pt first or
the old tokenizer and model will no longer match.

The text is cleaned while it is written: curly quotes and dashes become plain
ones and every other unusual symbol is dropped (about 96 distinct characters
remain). Datasets are streamed, so nothing huge is downloaded first.
"""
import os, re, sys
from pathlib import Path

from datasets import load_dataset

BASE = Path(__file__).resolve().parent
OUT = BASE / "train.txt"

if OUT.exists() and "--force" not in sys.argv:
    sys.exit("train.txt already exists. Run with --force to overwrite it.")

TARGET_MB = 200  # total size of train.txt

# (name, dataset id, config, split, how to get text, share of total)
SOURCES = [
    ("dialogue", "allenai/soda", None, "train",
     lambda r: "\n".join(r["dialogue"]), 0.30),
    ("science", "HuggingFaceTB/cosmopedia", "stanford", "train",
     lambda r: r["text"], 0.20),
    ("edu-web", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train",
     lambda r: r["text"], 0.25),
    ("wikipedia", "wikimedia/wikipedia", "20231101.en", "train",
     lambda r: r["text"], 0.25),
]

SEP = "\n\n"
PLAIN = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                       "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ", "\t": " "})


def clean(t):
    t = t.translate(PLAIN)
    t = re.sub(r"[^\n -~]", "", t)       # keep only newline and printable ASCII
    t = re.sub(r" {2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


tmp = OUT.with_suffix(".tmp")
with open(tmp, "w", encoding="utf-8") as out:
    for name, ds_id, config, split, get_text, share in SOURCES:
        budget = int(TARGET_MB * 1024 * 1024 * share)
        written = 0
        print(f"[{name}] collecting ~{budget // (1024 * 1024)} MB from {ds_id}", flush=True)
        ds = load_dataset(ds_id, config, split=split, streaming=True)
        for row in ds:
            text = clean(get_text(row))
            if len(text) < 200:          # skip tiny fragments
                continue
            out.write(text + SEP)
            written += len(text)
            if written >= budget:
                break
        print(f"[{name}] done, {written // (1024 * 1024)} MB", flush=True)
os.replace(tmp, OUT)
print("Finished: train.txt")
