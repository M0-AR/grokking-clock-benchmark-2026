"""Part 1 — The task: clock arithmetic table, tokens, train/test split.

P=113 clock: (a+b) mod P. 113*113 = 12769 problems.
Tokens: 0..112 numbers, 113 '='. Problem = (a, b, =) -> c.
Split: shuffle, keep `frac` for train, rest for test.
No structure leaks: tokens are labels; network never sees decimal order.
"""
from __future__ import annotations
import numpy as np


def build_table(P: int = 113, op: str = "add"):
    a = np.arange(P, dtype=np.int64)
    A, B = np.meshgrid(a, a, indexing="ij")
    if op == "add":
        C = (A + B) % P
    elif op == "sub":
        C = (A - B) % P
    elif op == "mul":
        C = (A * B) % P
    else:
        raise ValueError(op)
    return A.reshape(-1), B.reshape(-1), C.reshape(-1)


def split(P=113, frac=0.3, seed=0, op="add"):
    rng = np.random.default_rng(seed)
    A, B, C = build_table(P, op)
    n = len(A)
    idx = rng.permutation(n)
    n_train = int(n * frac)
    tr, te = idx[:n_train], idx[n_train:]
    eq = 113 if P == 113 else P  # '=' id = P
    def pack(ii):
        ta = A[ii].astype(np.int64)
        tb = B[ii].astype(np.int64)
        toks = np.stack([ta, tb, np.full_like(ta, eq)], axis=1)
        return toks, C[ii].astype(np.int64)
    train_toks, train_ans = pack(tr)
    test_toks, test_ans = pack(te)
    return (train_toks, train_ans), (test_toks, test_ans)


if __name__ == "__main__":
    (tr_t, tr_a), (te_t, te_a) = split()
    print(f"12769 problems, {len(tr_a)} train, {len(te_a)} test")
    # video check: 100+50 mod 113 = 37
    print("100+50 mod 113 =", (100 + 50) % 113)
    # 37+99 = 23 ?
    print("37+99 mod 113 =", (37 + 99) % 113)
