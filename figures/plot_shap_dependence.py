"""SHAP dependence plots: feature value vs SHAP value for top features"""
import sys, time, numpy as np, torch
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
})

timestamp = time.strftime("%Y%m%d_%H%M%S")
SAVE_DIR = "/Users/arcadio/flory_huggins/figures"
MODEL_PATH = "/Users/arcadio/flory_huggins/models/best_model.pt"

from models.attention_net import ChiPredictor
from features.data_loader import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_data(batch_size=32)
pool_df = data["pool_df"]
feature_cols = data["feature_cols"]

X = torch.tensor(pool_df[feature_cols].values, dtype=torch.float32)
X_np = X.numpy()
y = pool_df["chi"].values

model = ChiPredictor().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

FEATURE_LABELS = [
    "Mw (Polymer)", "LogP (Polymer)", "HBA (Polymer)", "HBD (Polymer)",
    "TPSA (Polymer)", "RotBonds (Polymer)", "Rings (Polymer)", "ArRings (Polymer)",
    "MolMR (Polymer)", "BertzCT (Polymer)", "Chi0v (Polymer)", "Chi1v (Polymer)",
    "Mw (Solvent)", "LogP (Solvent)", "HBA (Solvent)", "HBD (Solvent)",
    "TPSA (Solvent)", "RotBonds (Solvent)", "Rings (Solvent)", "ArRings (Solvent)",
    "MolMR (Solvent)", "BertzCT (Solvent)", "Chi0v (Solvent)", "Chi1v (Solvent)",
]

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).unsqueeze(-1)

import shap
wrapped = ModelWrapper(model).to(device)
explainer = shap.GradientExplainer(wrapped, X[:100].to(device))
shap_values = explainer.shap_values(X[:200].to(device))
shap_values = np.array(shap_values).squeeze()

shap_mean = np.abs(shap_values).mean(axis=0)
top6_idx = np.argsort(shap_mean)[-6:][::-1]  # top 6

# ── Figure: 2×3 grid of dependence plots ──
fig, axes = plt.subplots(2, 3, figsize=(183/25.4, 100/25.4))
axes = axes.flatten()

for ax, fi in zip(axes, top6_idx):
    fvals = X_np[:200, fi]
    shaps = shap_values[:, fi]
    ax.scatter(fvals, shaps, s=10, c="#4c72b0", alpha=0.6, edgecolors="none")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel(FEATURE_LABELS[fi], fontsize=7)
    ax.set_ylabel("SHAP value", fontsize=7)

plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/shap_dependence_{timestamp}.svg", bbox_inches="tight")
fig.savefig(f"{SAVE_DIR}/shap_dependence_{timestamp}.png", dpi=300, bbox_inches="tight")
print(f"Saved: shap_dependence_{timestamp}.svg / .png")
print(f"Top 6 features: {[FEATURE_LABELS[i] for i in top6_idx]}")
