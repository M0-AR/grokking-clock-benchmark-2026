"""Part 7 — Sweeps + ablations + hidden-pattern mining.

- data-fraction sweep (groks sooner with more data)
- weight-decay sweep (wd=0 never; larger wd faster to a point)
- seed sweep (different grok times + different key speeds)
- subtraction control (harder: non-symmetric; needs more data)
- embedding ablations: keep-only-keys (stays 100%) vs remove-keys (-> ~1%)
- FVE fit of trained logits by clock formula
- FSD-lite: frequency synchronization precursor (mean resultant length across heads/neurons)
"""
from __future__ import annotations
import json
import numpy as np, torch
from .task import split
from .model import TinyClockTransformer
from .train import train
from .analysis import fourier_shares, key_speeds
from .clock import clock_predict, fit_fve


@torch.no_grad()
def full_logits(model, toks, device, bs=2048):
    outs = []
    for i in range(0, len(toks), bs):
        t = torch.from_numpy(toks[i:i+bs]).long().to(device)
        lg, _ = model(t)
        outs.append(lg.cpu().numpy())
    return np.concatenate(outs, axis=0)


def embedding_keys(model):
    WE = model.WE.detach().cpu().numpy()[:113]
    shares, norms, _ = fourier_shares(WE)
    ks = key_speeds(shares, 4)
    return shares, ks, WE


def eval_handwritten(P=113, speeds=(1, 8, 21)):
    (tr_t, tr_a), (te_t, te_a) = split(P, 0.3, 0, "add")
    A = np.concatenate([tr_t[:, 0], te_t[:, 0]])
    B = np.concatenate([tr_t[:, 1], te_t[:, 1]])
    Y = np.concatenate([tr_a, te_a])
    pred = clock_predict(A, B, P, speeds)
    return float((pred == Y).mean())


def quick_sweep(device=None):
    """Fast benchmark (short epochs) to verify directions without 12k-epoch cost."""
    res = {}
    for frac in [0.25, 0.3, 0.5]:
        _, h = train(frac=frac, seed=0, epochs=1500, wd=1.0, log_every=1500, device=device)
        res[f"frac{frac}"] = h[-1] if h else {}
    for wd in [0.0, 0.5, 2.0]:
        _, h = train(frac=0.3, seed=0, epochs=1500, wd=wd, log_every=1500, device=device)
        res[f"wd{wd}"] = h[-1] if h else {}
    return res
