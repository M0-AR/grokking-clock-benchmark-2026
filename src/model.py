"""Part 2 — The model: 1-layer transformer written by hand, no ready-made layers.

Steps: look up (embed+pos) -> gather (4-head attention at '=') -> think (512 ReLU) -> score (113 logits).
Only torch tensors + Parameters. No nn.Transformer / nn.MultiheadAttention / nn.LayerNorm.
Sizes: d=128, heads=4 (d_head=32), mlp=512, P=113, vocab=114.
Params ~= 226k (matches video count order).
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


class TinyClockTransformer(torch.nn.Module):
    def __init__(self, P=113, d=128, heads=4, mlp=512, seed=0):
        super().__init__()
        self.P, self.d, self.heads, self.mlp_h = P, d, heads, mlp
        self.dh = d // heads
        self.vocab = P + 1  # extra '=' token id P
        g = torch.Generator().manual_seed(seed)
        def p(*s):
            return torch.nn.Parameter(torch.randn(*s, generator=g) * 0.2)
        self.WE = p(self.vocab, d)
        self.Wpos = p(3, d)
        self.WQ = p(heads, d, self.dh)
        self.WK = p(heads, d, self.dh)
        self.WV = p(heads, d, self.dh)
        self.WO = p(heads, self.dh, d)
        self.Win = p(mlp, d)
        self.bin = torch.nn.Parameter(torch.zeros(mlp))
        self.Wout = p(d, mlp)
        self.bout = torch.nn.Parameter(torch.zeros(d))
        self.WU = p(P, d)

    def forward(self, toks):
        # toks: (B,3) long
        B = toks.shape[0]
        x = self.WE[toks] + self.Wpos.unsqueeze(0)  # B,3,d
        # attention: query from pos2 only
        q = torch.einsum("bd,hde->bhe", x[:, 2, :], self.WQ)      # B,h,dh
        # k_bht = x_bt @ WK_h
        k = torch.stack([x @ self.WK[h] for h in range(self.heads)], dim=1)  # B,h,3,dh
        v = torch.stack([x @ self.WV[h] for h in range(self.heads)], dim=1)
        scores = torch.einsum("bhe,bhte->bht", q, k) / math.sqrt(self.dh)  # B,h,3
        w = F.softmax(scores, dim=-1)  # B,h,3
        blend = torch.einsum("bht,bhte->bhe", w, v)  # B,h,dh
        heads_out = torch.stack([blend[:, h, :] @ self.WO[h] for h in range(self.heads)], dim=1).sum(dim=1)  # B,d
        note = x[:, 2, :] + heads_out  # residual
        h = torch.relu(note @ self.Win.t() + self.bin)  # B,mlp
        note2 = note + h @ self.Wout.t() + self.bout
        logits = note2 @ self.WU.t()  # B,P
        cache = {"attn": w.detach(), "hidden": h.detach(), "note": note.detach()}
        return logits, cache

    def weight_size(self):
        return sum((p.detach().float() ** 2).sum().item() for p in self.parameters())

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = TinyClockTransformer()
    print("params:", m.count_params())
    toks = torch.tensor([[37, 99, 113]])
    logits, cache = m(toks)
    print("guess untrained:", int(logits.argmax(-1)), "true:", (37 + 99) % 113)
