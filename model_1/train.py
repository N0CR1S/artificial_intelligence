"""Train (or keep training) model 1, the small character-level model.

Run from anywhere:   python train.py
Options:             --steps 3000   number of training steps (default 3000)
                     --fresh        ignore model_1.pt and start a new model

Reads train.txt, saves model_1.pt. If model_1.pt already exists, training
continues from it. model_backup.pt is created once (only if missing) and
never overwritten, so it stays a safe copy.
"""
import argparse, os, shutil, sys, time
from pathlib import Path

import torch

from model import GPT, BLOCK_SIZE

BASE = Path(__file__).resolve().parent
TEXT, MODEL, BACKUP = BASE / "train.txt", BASE / "model_1.pt", BASE / "model_backup.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--fresh", action="store_true")
args = parser.parse_args()

# ---- settings ----
batch_size = 32
dropout = 0.1
lr_fresh, lr_resume = 1e-3, 5e-4
eval_every = 250

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- data ----
if not TEXT.exists():
    sys.exit(f"Missing {TEXT}. Put your training text there (see prepare.py).")
if TEXT.stat().st_size > 20_000_000:
    print("Warning: train.txt is very large for this small CPU model. "
          "It should be about 1 MB (the books). Is this the right file?")
text = TEXT.read_text(encoding="utf-8")
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"train.txt: {len(text):,} characters, {vocab_size} unique")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(20)
        for k in range(20):
            x, y = get_batch(split)
            losses[k] = model(x, y)[1].item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save():
    tmp = MODEL.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "chars": chars}, tmp)
    os.replace(tmp, MODEL)


# ---- model (continue from model_1.pt if possible) ----
model = GPT(vocab_size, dropout).to(device)
lr = lr_fresh
if MODEL.exists():
    ck = torch.load(MODEL, map_location=device, weights_only=True)
    if not BACKUP.exists():
        shutil.copy(MODEL, BACKUP)
        print("Created model_backup.pt")
    if args.fresh:
        shutil.copy(MODEL, BASE / "model_1_old.pt")
        print("--fresh: old model kept as model_1_old.pt, starting a new one")
    elif ck["chars"] != chars:
        sys.exit("The characters in train.txt differ from the ones model_1.pt was "
                 "trained on, so it cannot continue.\n"
                 "Use the original text, or run with --fresh to start a new model "
                 "(the old one is kept).")
    else:
        model.load_state_dict(ck["model"])
        lr = lr_resume
        print("Continuing from model_1.pt")
print("Parameters:", sum(p.numel() for p in model.parameters()), "| device:", device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)

# ---- training ----
start = time.time()
for it in range(args.steps + 1):
    if it % eval_every == 0 or it == args.steps:
        l = estimate_loss()
        print(f"step {it}: train {l['train']:.3f}  val {l['val']:.3f}  ({time.time() - start:.0f}s)",
              flush=True)
        if it > 0:
            save()
    if it == args.steps:
        break
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

print("Saved model_1.pt")
ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(("".join(itos[i] for i in model.generate(ctx, 600)[0].tolist())))
