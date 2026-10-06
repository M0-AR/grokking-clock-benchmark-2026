"""Plots from results/*.json (no retraining). Verifies histories exist and renders curves."""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
files = sorted(glob.glob("results/*.json"))
print("histories:", files)
for fp in files:
    try:
        j = json.load(open(fp))
    except Exception as e:
        print(fp, "skip", e); continue
    h = j.get("hist", [])
    if not h: continue
    ep = [r["epoch"] for r in h]
    plt.figure()
    plt.plot(ep, [r["train_acc"] for r in h], label="train_acc")
    plt.plot(ep, [r["test_acc"] for r in h], label="test_acc")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("acc"); plt.title(fp)
    out = fp.replace(".json", ".png")
    plt.savefig(out, dpi=120); plt.close()
    print("wrote", out)
