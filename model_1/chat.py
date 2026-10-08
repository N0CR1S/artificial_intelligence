"""Talk to model 1: type the start of a sentence, it continues the text.

Run from anywhere:   python chat.py
Commands:            /temp 0.5   lower = safer, higher = wilder (default 0.8)
                     /len 400    how many characters to write (default 400)
Empty line = quit.
"""
import sys
from pathlib import Path

import torch

from model import GPT, BLOCK_SIZE

sys.stdout.reconfigure(errors="replace")   # odd characters never crash the console
BASE = Path(__file__).resolve().parent

ck = torch.load(BASE / "model_1.pt", map_location="cpu", weights_only=True)
chars = ck["chars"]
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

model = GPT(len(chars), dropout=0.0)
model.load_state_dict(ck["model"])
model.eval()

temp, length = 0.8, 400
print("Model 1 loaded. Type a start of a sentence (empty line = quit).")
while True:
    p = input("\n> ")
    if not p:
        break
    if p.startswith("/temp "):
        temp = float(p.split()[1]); print("temperature =", temp); continue
    if p.startswith("/len "):
        length = int(p.split()[1]); print("length =", length); continue
    ids = [stoi[c] for c in p if c in stoi] or [0]
    out = model.generate(torch.tensor([ids]), length, temperature=temp)
    print("".join(itos[i] for i in out[0].tolist()))
