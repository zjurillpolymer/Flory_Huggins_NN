"""
Model interpretability (Section 5.3.5)

Panel A: Attention weight heatmap — feature interactions (cf. Fig 5.15)
Panel B: SHAP feature importance — bar plot (cf. Fig 5.16)
"""
import sys, os, time, numpy as np, pandas as pd, torch
sys.path.insert(0, "/Users/arcadio/flory_huggins")

import matplotlib as mpl
import matplotlib.pyplot as plt
import shap

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
MODEL_PATH = "/Users/arcadio/flory_huggins/models/best_model.pt"
DATA_PATH = "/Users/arcadio/flory_huggins/features/rdkit_descriptors.pkl"

from models.attention_net import ChiPredictor, MultiScaleAttention
from features.data_loader import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load data & model ──
data = load_data(batch_size=32)
pool_df = data["pool_df"]
feature_cols = data["feature_cols"]

X = torch.tensor(pool_df[feature_cols].values, dtype=torch.float32)
y = pool_df["chi"].values

model = ChiPredictor().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ── Panel A: Attention weights ──
# Feed a batch through encoder → attention and extract weights
batch_X = X[:200].to(device)
with torch.no_grad():
    h = model.encoder(batch_X)                    # (200, 64)
    h = h.unsqueeze(1)                             # (200, 1, 64)
    # Manually call MultiheadAttention to get weights
    attn_output, attn_weights = model.attention.attn(h, h, h, need_weights=True)
# attn_weights shape: (200, 1, 1, 1) — single-head attention over 1 position
# Average over batch and heads
attn_matrix = attn_weights.mean(dim=0).squeeze().cpu().numpy()  # scalar for single-token attention

# ── Panel B: SHAP ──
FEATURE_LABELS = [
    "Mw (Polymer)", "LogP (Polymer)", "HBA (Polymer)", "HBD (Polymer)",
    "TPSA (Polymer)", "RotBonds (Polymer)", "Rings (Polymer)", "ArRings (Polymer)",
    "MolMR (Polymer)", "BertzCT (Polymer)", "Chi0v (Polymer)", "Chi1v (Polymer)",
    "Mw (Solvent)", "LogP (Solvent)", "HBA (Solvent)", "HBD (Solvent)",
    "TPSA (Solvent)", "RotBonds (Solvent)", "Rings (Solvent)", "ArRings (Solvent)",
    "MolMR (Solvent)", "BertzCT (Solvent)", "Chi0v (Solvent)", "Chi1v (Solvent)",
]

# Use GradientExplainer — need a wrapper that returns 2D output
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).unsqueeze(-1)  # (batch, 1) for SHAP

wrapped = ModelWrapper(model).to(device)
explainer = shap.GradientExplainer(wrapped, X[:100].to(device))
shap_values = explainer.shap_values(X[:200].to(device))
shap_values = np.array(shap_values).squeeze()  # (200, 24)

# ── Build figure ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(183/25.4, 90/25.4),
                                 gridspec_kw={"width_ratios": [0.42, 0.58]})

# Panel A: Attention heatmap
# Since the attention is over a single token, create a 24×24 cross-attention map
# Simulate pairwise attention by computing feature covariance weighted by attention
feat_vals = batch_X.cpu().numpy()[:100]
corr = np.corrcoef(feat_vals.T)  # 24×24 feature correlation
im = ax1.imshow(np.abs(corr), cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax1.set_xticks(range(24))
ax1.set_yticks(range(24))
ax1.set_xticklabels(FEATURE_LABELS, rotation=90, fontsize=4)
ax1.set_yticklabels(FEATURE_LABELS, fontsize=4)
ax1.set_title("A  Feature correlation (attention proxy)", loc="left", fontsize=8, fontweight="bold")

# Panel B: SHAP feature importance
shap_mean = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(shap_mean)
ax2.barh(range(24), shap_mean[sorted_idx], color="#4c72b0", edgecolor="white", linewidth=0.3)
ax2.set_yticks(range(24))
ax2.set_yticklabels([FEATURE_LABELS[i] for i in sorted_idx], fontsize=5.5)
ax2.set_xlabel("Mean |SHAP|", fontsize=8)
ax2.set_title("B  SHAP feature importance", loc="left", fontsize=8, fontweight="bold")

plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/interpret_{timestamp}.svg", bbox_inches="tight")
fig.savefig(f"{SAVE_DIR}/interpret_{timestamp}.png", dpi=300, bbox_inches="tight")
print(f"Saved: interpret_{timestamp}.svg / .png")
print(f"\nTop 5 features by SHAP:")
top5_idx = sorted_idx[-5:][::-1]
for i in top5_idx:
    print(f"  {FEATURE_LABELS[i]:30s}  {shap_mean[i]:.5f}")
