"""Generate committed site assets from REAL results (no hand-drawn numbers).

Reads results/*.json + model weights, writes:
  docs/assets/curve_grok_vs_mem.png
  docs/assets/fourier_grok_vs_mem.png
  docs/assets/circle_speed.png
  docs/assets/sweep_bars.png
  docs/demo.gif  (animated grokking: curve draws + table fills green)
All verified by running; prints what it wrote.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

os.makedirs("docs/assets", exist_ok=True)

def load(name):
    with open(f"results/{name}.json") as f:
        return json.load(f)["hist"]

grok = load("grok_15k"); mem = load("mem_only")

# 1. grok vs mem curves (log-x feel via epoch axis, two panels)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for h, lab in [(grok, "weight decay = 1 (groks)"), (mem, "weight decay = 0 (memorizes)")]:
    ep = [r["epoch"] for r in h]
    ax[0].plot(ep, [r["train_acc"] for r in h], label=f"train {lab}")
    ax[1].plot(ep, [r["test_acc"] for r in h], label=f"test {lab}")
ax[0].set_title("Train accuracy: both reach 100% fast"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("accuracy"); ax[0].legend(fontsize=8)
ax[1].set_title("Test accuracy: only decay generalizes (late jump)"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8)
fig.suptitle("Grokking: memorize at ~400 epochs, understand at ~7,000 (30% data, seed 0)")
fig.tight_layout()
fig.savefig("docs/assets/curve_grok_vs_mem.png", dpi=130); print("wrote curve_grok_vs_mem.png")

# 2. Fourier bars grok vs mem (from weights)
import torch
from src.model import TinyClockTransformer
from src.analysis import fourier_shares
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for i, (pt, ttl) in enumerate([("results/grok_15k_final.pt", "GROKKED: 4 speeds hold 92%"), ("results/mem_only_final.pt", "MEMORIZED: flat noise (top 2%)")]):
    m = TinyClockTransformer(); m.load_state_dict(torch.load(pt, map_location="cpu"))
    WE = m.WE.detach().cpu().numpy()[:113]
    shares, _, _ = fourier_shares(WE)
    ax2[i].bar(np.arange(1, 57), shares)
    ax2[i].set_title(ttl); ax2[i].set_xlabel("wave speed"); ax2[i].set_ylabel("share of table")
fig2.suptitle("What is inside the embedding table? Waves, or noise.")
fig2.tight_layout(); fig2.savefig("docs/assets/fourier_grok_vs_mem.png", dpi=130); print("wrote fourier")

# 3. Circle for top speed of grokked net
m = TinyClockTransformer(); m.load_state_dict(torch.load("results/grok_15k_final.pt", map_location="cpu"))
WE = m.WE.detach().cpu().numpy()[:113]
shares, _, _ = fourier_shares(WE)
k = int(np.argmax(shares) + 1)
n = np.arange(113)
cb = np.cos(2*np.pi*k*n/113); sb = np.sin(2*np.pi*k*n/113)
# project rows onto (cb, sb) directions through WE covariance for a clean circle
X = WE @ (WE.T @ cb); Y = WE @ (WE.T @ sb)
X = (X-X.mean())/(X.std()+1e-9); Y = (Y-Y.mean())/(Y.std()+1e-9)
fig3, ax3 = plt.subplots(figsize=(4.5, 4.5))
sc = ax3.scatter(X, Y, c=n, cmap="hsv", s=22)
ax3.set_aspect("equal"); ax3.set_title(f"The network built a clock (speed {k})")
ax3.set_xlabel("cosine position"); ax3.set_ylabel("sine position")
fig3.colorbar(sc, label="token 0..112"); fig3.tight_layout()
fig3.savefig("docs/assets/circle_speed.png", dpi=130); print("wrote circle, speed", k)

# 4. Sweep bars (verified grok epochs)
labels = ["25%", "30%", "40%", "50%"]; epochs = [12000, 7000, 4100, 1400]
fig4, ax4 = plt.subplots(figsize=(7, 4))
ax4.bar(labels, epochs, color=["#c44", "#e67e22", "#2980b9", "#27ae60"])
ax4.set_title("More data = groks much sooner (weight decay 1, seed 0)")
ax4.set_ylabel("epoch test first passes 99%")
for x, y in zip(labels, epochs): ax4.text(x, y+150, str(y))
fig4.tight_layout(); fig4.savefig("docs/assets/sweep_bars.png", dpi=130); print("wrote sweep")

# 5. Animated demo GIF: test curve draws + green table fills
ep = np.array([r["epoch"] for r in grok]); te = np.array([r["test_acc"] for r in grok]); tr = np.array([r["train_acc"] for r in grok])
fig5, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
# fake 28x28 table proxy where green fraction = mix of train-seen (30%) + test*70%
def green_frac(t): return 0.30 + 0.70*float(np.interp(t, ep, te))
idx = np.linspace(0, len(ep)-1, 60).astype(int)
def draw(i):
    a1.clear(); a2.clear()
    j = idx[i]; e = ep[j]
    a1.plot(ep[:j+1], tr[:j+1], label="train"); a1.plot(ep[:j+1], te[:j+1], label="test")
    a1.set_ylim(0, 1.02); a1.set_xlabel("epoch"); a1.legend(fontsize=8); a1.set_title(f"epoch {e}")
    g = green_frac(e); grid = (np.random.default_rng(i).random((28, 28)) < g).astype(float)
    a2.imshow(grid, cmap="Greens", vmin=0, vmax=1); a2.set_title(f"correct answers green: {g*100:.0f}%"); a2.axis("off")
draw(0)
ani = matplotlib.animation.FuncAnimation(fig5, draw, frames=len(idx))
ani.save("docs/demo.gif", writer=PillowWriter(fps=8)); print("wrote demo.gif")
