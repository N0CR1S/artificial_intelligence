"""Model 1: small character-level GPT (about 0.8 million parameters).

Used by train.py and chat.py. The layer names must not change, otherwise the
saved model_1.pt can no longer be loaded.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

BLOCK_SIZE = 128          # how many characters the model sees at once
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4


class Head(nn.Module):
    def __init__(self, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        w = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = self.drop(F.softmax(w, dim=-1))
        return w @ v


class MultiHead(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(N_EMBD // N_HEAD, dropout) for _ in range(N_HEAD)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class Block(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.sa = MultiHead(dropout)
        self.ff = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.ReLU(),
                                nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(dropout))
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok = nn.Embedding(vocab_size, N_EMBD)
        self.pos = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(dropout) for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.ln(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8):
        for _ in range(n_new):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
