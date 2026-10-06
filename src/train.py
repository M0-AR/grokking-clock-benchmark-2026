"""Part 3+4 — Training: guess -> measure (CE, float64) -> nudge (AdamW). Plus weight decay.

One epoch = full batch over all train problems (3830 at 30%).
Logs every 50 epochs: train acc, test acc, train loss, weight_size = sum w^2.
Weight decay: AdamW wd param (wd=1 default groks; wd=0 memorizes only).
Loss in float64 for stability at 1e-7 (video note).
"""
from __future__ import annotations
import argparse, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from .task import split
from .model import TinyClockTransformer


@torch.no_grad()
def accuracy(model, toks, ans, device, bs=2048):
    ok = 0
    for i in range(0, len(toks), bs):
        t = torch.from_numpy(toks[i:i+bs]).long().to(device)
        logits, _ = model(t)
        ok += (logits.argmax(-1).cpu().numpy() == ans[i:i+bs]).sum()
    return float(ok) / len(ans)


def train(P=113, frac=0.3, seed=0, op="add", epochs=6000, lr=1e-3, wd=1.0,
          log_every=100, device=None, ckpt_every=0, out=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    (tr_t, tr_a), (te_t, te_a) = split(P, frac, seed, op)
    model = TinyClockTransformer(P=P, seed=seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                            betas=(0.9, 0.98))
    # linear warmup 10 steps (paper)
    def set_lr(e):
        f = min(1.0, (e + 1) / 10.0)
        for g in opt.param_groups:
            g["lr"] = lr * f
    tr_t_t = torch.from_numpy(tr_t).long().to(device)
    tr_a_t = torch.from_numpy(tr_a).long().to(device)
    hist = []
    t0 = time.time()
    for e in range(1, epochs + 1):
        set_lr(e)
        model.train()
        opt.zero_grad()
        logits, _ = model(tr_t_t)
        loss = F.cross_entropy(logits.double(), tr_a_t).float()
        loss.backward()
        opt.step()
        if e % log_every == 0 or e == 1:
            model.eval()
            with torch.no_grad():
                tra = accuracy(model, tr_t, tr_a, device)
                tea = accuracy(model, te_t, te_a, device)
            ws = model.weight_size()
            row = {"epoch": e, "train_loss": float(loss.item()),
                   "train_acc": tra, "test_acc": tea, "wsize": ws,
                   "elapsed": time.time() - t0}
            hist.append(row)
            print(row, flush=True)
            if out and ckpt_every and e % ckpt_every == 0:
                torch.save(model.state_dict(), out.replace(".json", f"_e{e}.pt"))
    if out:
        with open(out, "w") as f:
            json.dump({"cfg": dict(P=P, frac=frac, seed=seed, op=op,
                                   epochs=epochs, lr=lr, wd=wd), "hist": hist}, f)
        torch.save(model.state_dict(), out.replace(".json", "_final.pt"))
    return model, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--op", default="add")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    train(epochs=a.epochs, wd=a.wd, frac=a.frac, seed=a.seed, op=a.op, out=a.out)
