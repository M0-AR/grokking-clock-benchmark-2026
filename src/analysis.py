"""Part 5 — Looking inside: Fourier anatomy, Gini, restricted/excluded progress.

- col DFT: WE[numbers,:] (113x128) -> per-frequency share (56 speeds).
- key speeds: top-k by share (default 4).
- circle plot data: (cos,sin) pair per speed -> dots in order.
- Gini of Fourier norms (sparsity, Nanda §5.1).
- restricted loss (keep key freqs only) / excluded loss (remove key freqs).
- neuron maps: how strongly each MLP neuron fires per (a,b).
"""
from __future__ import annotations
import numpy as np


def fourier_shares(WE_numbers: np.ndarray):
    """WE_numbers: (P,d). Returns shares[1..56], norms, full complex comps."""
    P, d = WE_numbers.shape
    F = np.fft.rfft(WE_numbers, axis=0)  # (P//2+1, d)
    norms = np.linalg.norm(F, axis=1)  # includes k=0 DC
    tot = (norms[1:] ** 2).sum()
    shares = (norms[1:] ** 2) / max(tot, 1e-30)  # len 56 for P=113
    return shares, norms, F


def key_speeds(shares, k=4):
    return list(np.argsort(shares)[::-1][:k] + 1)  # 1-indexed speeds


def gini(x):
    x = np.asarray(x, dtype=float).ravel()
    x = np.abs(x)
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def circle_xy(WE_numbers, speed):
    P = WE_numbers.shape[0]
    F = np.fft.rfft(WE_numbers, axis=0)
    k = int(speed)
    # reconstruct rank-2 circle for frequency k: Re/Im projections
    comp = F[k]  # (d,) complex
    # project each row onto cos/sin basis via least squares: use FFT basis vectors
    n = np.arange(P)
    cb = np.cos(2 * np.pi * k * n / P)
    sb = np.sin(2 * np.pi * k * n / P)
    # coordinates: dot with normalized basis
    X = WE_numbers @ (WE_numbers.T @ cb)
    Y = WE_numbers @ (WE_numbers.T @ sb)
    # simpler + stable: use DFT coefficients directly for angle ordering:
    # angle of each token = arg of its projection onto F[k] direction
    proj = WE_numbers @ comp.conj()
    ang = np.angle(proj)
    order = np.argsort(ang)
    return ang, order


def keep_only_speeds(WE_numbers, speeds):
    P, d = WE_numbers.shape
    F = np.fft.rfft(WE_numbers, axis=0)
    mask = np.zeros(F.shape[0], dtype=bool)
    mask[0] = True
    mask[list(speeds)] = True
    F2 = np.where(mask[:, None], F, 0)
    return np.fft.irfft(F2, n=P, axis=0)


def remove_speeds(WE_numbers, speeds):
    P, d = WE_numbers.shape
    F = np.fft.rfft(WE_numbers, axis=0)
    F2 = F.copy()
    F2[list(speeds)] = 0
    return np.fft.irfft(F2, n=P, axis=0)
