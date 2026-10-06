"""Progress measures (Nanda §5): restricted/excluded loss + FSD-lite + Gini + |w|^2."""
from __future__ import annotations
import numpy as np, torch
import torch.nn.functional as F


@torch.no_grad()
def logits_all(model, toks, device, bs=4096):
    outs = []
    for i in range(0, len(toks), bs):
        t = torch.from_numpy(toks[i:i+bs]).long().to(device)
        lg, _ = model(t)
        outs.append(lg.cpu().numpy())
    return np.concatenate(outs, 0)
