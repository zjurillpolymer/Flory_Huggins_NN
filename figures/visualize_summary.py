"""
Figure: Current progress summary — χ prediction with uncertainty quantification

Panels:
  A: Method comparison — MAE and R² for Baseline / MC Dropout / Ensemble
  B: Predicted vs actual χ scatter (MC Dropout, best-performing method)
  C: Uncertainty vs |error| scatter, comparing MC Dropout vs Ensemble
"""

import sys, os, time, numpy as np, torch
sys.path.insert(0, "/Users/arcadio/flory_huggins")

from models.attention_net import ChiPredictor
from features.data_loader import load_data, df_to_loader
from models.uncertainty import mc_dropout, ensemble_inference
from sklearn.metrics import mean_absolute_error, r2_score

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

# ── 1. Load data ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = load_data(batch_size=8)
pool_df = data["pool_df"].sample(frac=1, random_state=42)
n_val = int(len(pool_df) * 0.2)
train_loader = df_to_loader(pool_df[:-n_val], shuffle=True)
val_loader = df_to_loader(pool_df[-n_val:])

y_true = np.concatenate([y.numpy() for _, y in val_loader])

# ── 2. Run all methods ──
print("Running MC Dropout ...")
mc_mean, mc_unc = mc_dropout(val_loader, num_samples=50)

print("Running Ensemble inference ...")
en_mean, en_unc = ensemble_inference(val_loader, num_models=4)

# ── 3. Compute metrics ──
baseline_mae, baseline_r2 = 0.244, 0.643  # from previous run

mc_mae = mean_absolute_error(y_true, mc_mean)
mc_r2 = r2_score(y_true, mc_mean)
en_mae = mean_absolute_error(y_true, en_mean)
en_r2 = r2_score(y_true, en_mean)

mc_err = np.abs(y_true - mc_mean)
en_err = np.abs(y_true - en_mean)
mc_corr = np.corrcoef(mc_unc, mc_err)[0, 1]
en_corr = np.corrcoef(en_unc, en_err)[0, 1]

print(f"Baseline:  MAE={baseline_mae:.3f}, R²={baseline_r2:.3f}")
print(f"MC Dropout: MAE={mc_mae:.3f}, R²={mc_r2:.3f}, r(unc,err)={mc_corr:.3f}")
print(f"Ensemble:  MAE={en_mae:.3f}, R²={en_r2:.3f}, r(unc,err)={en_corr:.3f}")

# ── 4. Build figure ──
fig = plt.figure(figsize=(183 / 25.4, 100 / 25.4))  # ~7.2 × 3.9 inches

# Panel A: MAE + R² bar chart
ax1 = fig.add_axes([0.06, 0.15, 0.28, 0.78])
x = np.arange(3)
width = 0.32
mae_vals = [baseline_mae, mc_mae, en_mae]
r2_vals = [baseline_r2, mc_r2, en_r2]
labels = ["Baseline", "MC\nDropout", "Ensemble"]
colors = ["#b0b0b0", "#4c72b0", "#c44e52"]

bars1 = ax1.bar(x - width / 2, mae_vals, width, color=colors, edgecolor="white", linewidth=0.3, zorder=2)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=7)
ax1.set_ylabel("MAE", fontsize=8)
ax1.set_ylim(0, 0.35)
for bar, v in zip(bars1, mae_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

ax1b = ax1.twinx()
ax1b.spines["right"].set_visible(True)
ax1b.spines["right"].set_position(("axes", 1.0))
ax1b.spines["right"].set_linewidth(0.8)
bars2 = ax1b.bar(x + width / 2, r2_vals, width, color=colors, edgecolor="white", linewidth=0.3, alpha=0.5, zorder=2)
ax1b.set_ylabel("R²", fontsize=8)
ax1b.set_ylim(0, 1.05)
for bar, v in zip(bars2, r2_vals):
    ax1b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
              f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
ax1.set_title("A  Method comparison", loc="left", fontsize=8, fontweight="bold")

# Panel B: Predicted vs Actual (MC Dropout)
ax2 = fig.add_axes([0.40, 0.15, 0.28, 0.78])
ax2.scatter(y_true, mc_mean, s=8, c="#4c72b0", alpha=0.5, edgecolors="none")
lims = [min(y_true.min(), mc_mean.min()) - 0.2, max(y_true.max(), mc_mean.max()) + 0.2]
ax2.plot(lims, lims, "k--", linewidth=0.6)
ax2.set_xlim(lims)
ax2.set_ylim(lims)
ax2.set_xlabel("True χ", fontsize=8)
ax2.set_ylabel("Predicted χ", fontsize=8)
ax2.text(0.95, 0.05, f"MAE = {mc_mae:.3f}\nR² = {mc_r2:.3f}",
         transform=ax2.transAxes, fontsize=7, va="bottom", ha="right")
ax2.set_title("B  MC Dropout prediction", loc="left", fontsize=8, fontweight="bold")

# Panel C: Uncertainty vs |error|
ax3 = fig.add_axes([0.74, 0.15, 0.24, 0.78])
ax3.scatter(mc_err, mc_unc, s=6, c="#4c72b0", alpha=0.5, label=f"MC Dropout (r={mc_corr:.2f})", edgecolors="none")
ax3.scatter(en_err, en_unc, s=6, c="#c44e52", alpha=0.5, label=f"Ensemble (r={en_corr:.2f})", edgecolors="none")
ax3.set_xlabel("|Error|", fontsize=8)
ax3.set_ylabel("Uncertainty", fontsize=8)
ax3.legend(fontsize=6.5, markerscale=1.5, loc="upper left")
ax3.set_title("C  Uncertainty vs Error", loc="left", fontsize=8, fontweight="bold")

fig.savefig(f"{SAVE_DIR}/summary_{timestamp}.svg", bbox_inches="tight")
fig.savefig(f"{SAVE_DIR}/summary_{timestamp}.png", dpi=300, bbox_inches="tight")
print(f"\nSaved: summary_{timestamp}.svg / .png")
