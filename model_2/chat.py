"""Talk to model 2: type a start, it continues the text.

Run from anywhere:   python chat.py
Commands:            /temp 0.5   lower = safer, higher = wilder (default 0.8)
                     /len 150    how many tokens to write (default 150)
Empty line = quit.

This is the base model: it continues text, it does not answer yet.
"""
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT

sys.stdout.reconfigure(errors="replace")   # odd characters never crash the console
BASE = Path(__file__).resolve().parent

tok = Tokenizer.from_file(str(BASE / "tok.json"))
model = GPT(dropout=0.0)
model.load_state_dict(torch.load(BASE / "model_2.pt", map_location="cpu", weights_only=True))
model.eval()

temp, length = 0.8, 150
print("Model 2 loaded. Type a start (empty line = quit).")
while True:
    p = input("\n> ")
    if not p:
        break
    if p.startswith("/temp "):
        temp = float(p.split()[1]); print("temperature =", temp); continue
    if p.startswith("/len "):
        length = int(p.split()[1]); print("length =", length); continue
    ids = tok.encode(p).ids or [0]
    out = model.generate(torch.tensor([ids]), length, temperature=temp)
    print(tok.decode(out[0].tolist()))
