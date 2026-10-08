"""Train (or keep training) model 2. Needs a GPU to be fast (for example Colab).

Run from anywhere:   python train.py
Options:             --steps 6000   number of training steps (default 6000)
                     --batch 32     batch size (use 16 if the GPU runs out of memory)
                     --fresh        ignore model_2.pt and start a new model

What it does, in order (each step is skipped if its file already exists):
  1. tok.json     the tokenizer, trained on train.txt
  2. tokens.npy   train.txt converted to numbers (a cache, safe to delete;
                  rebuilt automatically when train.txt is newer)
  3. model_2.pt   training. If model_2.pt exists, training continues from it.
model_backup.pt is created once (only if missing) and never overwritten.
"""
import argparse, math, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer, Tokenizer

from model import GPT, BLOCK_SIZE, VOCAB_SIZE

BASE = Path(__file__).resolve().parent
TEXT, TOK, TOKENS = BASE / "train.txt", BASE / "tok.json", BASE / "tokens.npy"
MODEL, BACKUP = BASE / "model_2.pt", BASE / "model_backup.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--fresh", action="store_true")
args = parser.parse_args()

# ---- settings ----
dropout = 0.1
lr_fresh, lr_resume = 6e-4, 3e-4
eval_every = 500
eval_batches = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
if not use_amp:
    print("WARNING: no GPU found. Training on the CPU is very slow (hours per 1000 steps).")
torch.manual_seed(1337)
np.random.seed(1337)

# ---- 1. tokenizer ----
if not TEXT.exists():
    sys.exit(f"Missing {TEXT}")
if not TOK.exists():
    print("Training tokenizer (3-6 minutes)...")
    t = ByteLevelBPETokenizer()
    t.train(files=[str(TEXT)], vocab_size=VOCAB_SIZE, min_frequency=2)
    t.save(str(TOK))
tok = Tokenizer.from_file(str(TOK))

# ---- 2. text -> numbers ----
if not TOKENS.exists() or TOKENS.stat().st_mtime < TEXT.stat().st_mtime:
    print("Converting train.txt to tokens (5-10 minutes)...")
    chunks, batch = [], []

    def flush():
        global batch
        enc = tok.encode_batch(batch)
        chunks.append(np.fromiter((i for e in enc for i in e.ids), dtype=np.uint16))
        batch = []

    with open(TEXT, encoding="utf-8") as f:
        for line in f:
            batch.append(line)
            if len(batch) >= 100000:
                flush()
    if batch:
        flush()
    np.save(TOKENS, np.concatenate(chunks))
tokens = np.load(TOKENS)

# Validation = every 100th chunk spread over the whole file, so it contains
# dialogue AND encyclopedia text, like the training part.
chunk = int(min(50_000, max(1_000, len(tokens) // 20)))
pieces = [tokens[i:i + chunk] for i in range(0, len(tokens), chunk)]
val_data = np.concatenate([p for i, p in enumerate(pieces) if i % 100 == 0])
train_data = np.concatenate([p for i, p in enumerate(pieces) if i % 100 != 0])
print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")


def get_batch(d):
    ix = np.random.randint(0, len(d) - BLOCK_SIZE - 1, args.batch)
    x = np.stack([d[i:i + BLOCK_SIZE] for i in ix]).astype(np.int64)
    y = np.stack([d[i + 1:i + 1 + BLOCK_SIZE] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def save():
    tmp = MODEL.with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, MODEL)


# ---- 3. model ----
model = GPT(dropout).to(device)
max_lr, warmup = lr_fresh, 200
if MODEL.exists():
    if not BACKUP.exists():
        shutil.copy(MODEL, BACKUP)
        print("Created model_backup.pt")
    if args.fresh:
        shutil.copy(MODEL, BASE / "model_2_old.pt")
        print("--fresh: old model kept as model_2_old.pt, starting a new one")
    else:
        model.load_state_dict(torch.load(MODEL, map_location=device, weights_only=True))
        max_lr, warmup = lr_resume, 100
        print("Continuing from model_2.pt")
min_lr = max_lr / 10
print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}", "| device:", device)

opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler(enabled=use_amp)


def lr_at(it):
    if it < warmup:
        return max_lr * (it + 1) / warmup
    p = (it - warmup) / max(1, args.steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def evaluate():
    model.eval()
    out = {}
    for name, d in (("train", train_data), ("val", val_data)):
        ls = []
        for _ in range(eval_batches):
            x, y = get_batch(d)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                ls.append(model(x, y)[1].item())
        out[name] = sum(ls) / len(ls)
    model.train()
    return out


# ---- training ----
t0 = time.time()
model.train()
for it in range(args.steps + 1):
    if it % eval_every == 0 or it == args.steps:
        l = evaluate()
        print(f"step {it}: train {l['train']:.3f}  val {l['val']:.3f}  ({time.time() - t0:.0f}s)",
              flush=True)
        if it > 0:
            save()
    if it == args.steps:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    x, y = get_batch(train_data)
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

print("Saved model_2.pt")
model.eval()
ctx = torch.tensor([tok.encode("Hi, how are you?").ids], device=device)
print(tok.decode(model.generate(ctx, 150)[0].tolist()))
