"""Part 6 — The clock: handwritten cosine algorithm, no training, no weights.

score(c | a,b) = sum_k cos(w_k (a+b-c)), w_k = 2*pi*k/P.
Best = argmax. Runner-up analysis shows why multiple speeds sharpen the peak.
Also: fit trained logits with same formula -> FVE (Nanda §4.2: ~95%).
"""
from __future__ import annotations
import numpy as np


def clock_scores(a, b, P=113, speeds=(1, 8, 21)):
    c = np.arange(P)
    s = np.zeros(P)
    for k in speeds:
        s = s + np.cos(2 * np.pi * k * (a + b - c) / P)
    return s


def clock_predict(A, B, P=113, speeds=(1, 8, 21)):
    out = np.empty(len(A), dtype=int)
    for i, (a, b) in enumerate(zip(A, B)):
        out[i] = int(np.argmax(clock_scores(int(a), int(b), P, speeds)))
    return out


def fit_fve(logits_ab_c, A, B, P=113, speeds=(1, 8, 21)):
    """logits_ab_c: (N,P) trained logits. Fit alpha_k * cos(wk(a+b-c)). Returns FVE + alphas."""
    N = len(A)
    C = np.arange(P)
    X = np.stack([np.cos(2 * np.pi * k * (A[:, None] + B[:, None] - C[None, :]) / P)
                  for k in speeds], axis=-1)  # N,P,K
    Y = logits_ab_c
    # least squares per problem: solve X @ alpha = Y
    Xr = X.reshape(-1, len(speeds))
    Yr = Y.reshape(-1)
    alpha, *_ = np.linalg.lstsq(Xr, Yr, rcond=None)
    Yhat = X @ alpha
    ss = ((Y - Y.mean()) ** 2).sum()
    fve = 1 - ((Y - Yhat) ** 2).sum() / max(ss, 1e-30)
    return float(fve), alpha
