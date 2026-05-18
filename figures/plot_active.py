"""Figure: Active learning results — MC Dropout / Ensemble vs Random"""
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, "/Users/arcadio/flory_huggins")

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

timestamp = time.strftime("%Y%m%d_%H%M%S")
SAVE_DIR = "/Users/arcadio/flory_huggins/figures"

# Load results
random = pd.read_pickle("/Users/arcadio/flory_huggins/experiments/history_random.pkl")
mc = pd.read_pickle("/Users/arcadio/flory_huggins/experiments/history_mc.pkl")
en = pd.read_pickle("/Users/arcadio/flory_huggins/experiments/history_ensemble.pkl")

# ==== Figure: 2 panels ====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(183/25.4, 70/25.4))

colors = {"Random": "#b0b0b0", "MC Dropout": "#4c72b0", "Ensemble": "#c44e52"}

# Panel A: MAE learning curves
for label, hist, marker in [("Random", random, "o"), ("MC Dropout", mc, "s"), ("Ensemble", en, "^")]:
    ax1.plot(hist["n_labeled"], hist["mae"], label=label, color=colors[label],
             marker=marker, markersize=3, linewidth=0.8, alpha=0.8)
ax1.set_xlabel("Labeled samples", fontsize=8)
ax1.set_ylabel("MAE", fontsize=8)
ax1.legend(fontsize=7)
ax1.set_title("A  MAE vs labeled size", loc="left", fontsize=8, fontweight="bold")

# Panel B: R² learning curves
for label, hist in [("Random", random), ("MC Dropout", mc), ("Ensemble", en)]:
    ax2.plot(hist["n_labeled"], hist["r2"], label=label, color=colors[label],
             marker=marker, markersize=3, linewidth=0.8, alpha=0.8)
ax2.set_xlabel("Labeled samples", fontsize=8)
ax2.set_ylabel("R²", fontsize=8)
ax2.legend(fontsize=7)
ax2.set_title("B  R² vs labeled size", loc="left", fontsize=8, fontweight="bold")

plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/active_learning_{timestamp}.svg", bbox_inches="tight")
fig.savefig(f"{SAVE_DIR}/active_learning_{timestamp}.png", dpi=300, bbox_inches="tight")
print(f"Saved: active_learning_{timestamp}.svg / .png")
