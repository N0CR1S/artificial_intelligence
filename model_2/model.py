"""Model 2: GPT with a word-piece (BPE) tokenizer, about 25 million parameters.

Used by train.py and chat.py. The layer names must not change, otherwise the
saved model_2.pt can no longer be loaded.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

VOCAB_SIZE = 8192
BLOCK_SIZE = 256          # context length in tokens (about 1000 characters)
N_EMBD, N_HEAD, N_LAYER = 512, 8, 8


class Attn(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))


class Block(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.attn = Attn(dropout)
        self.ff = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
                                nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dropout) for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok.weight          # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8, top_k=50):
        for _ in range(n_new):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx
