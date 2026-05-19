"""SHAP analysis for SchNet QM properties (μ, α, HOMO, LUMO)"""
import sys, time, numpy as np, torch
sys.path.insert(0, "/Users/arcadio/flory_huggins")
sys.path.insert(0, "/Users/arcadio/Slimnet")
sys.path.insert(0, "/Users/arcadio/Slimnet/base_model_molecule_encoder")

from Schnet_model_monomer import Schnet_monomer
from features.schnet_loader import get_dataloaders
import shap
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
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── Build QM-only model ──
class QMChiModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Schnet_monomer(hidden_dim=128, n_layers=6)
        ckpt = torch.load("/Users/arcadio/flory_huggins/models/best_schnet.pt",
                           map_location="cpu")
        self.encoder.load_state_dict(ckpt)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(9, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 1),
        )
        # Load QM-only weights
        state = torch.load("/Users/arcadio/flory_huggins/models/best_qm.pt",
                            map_location="cpu")
        mlp_state = {k.replace("mlp.", ""): v for k, v in state.items() if k.startswith("mlp.")}
        self.mlp.load_state_dict(mlp_state)

    def forward(self, batch):
        qm_p = self.encoder(batch["poly_z"], batch["poly_pos"],
                             batch["poly_edge"], batch["poly_batch"])
        qm_s = self.encoder(batch["solv_z"], batch["solv_pos"],
                             batch["solv_edge"], batch["solv_batch"])
        x = torch.cat([qm_p, qm_s, batch["temp"].unsqueeze(-1)], dim=-1)
        return self.mlp(x).squeeze(-1)

# ── Get data ──
_, _, test_loader = get_dataloaders(batch_size=16)
batch = next(iter(test_loader))
model = QMChiModel().to(device)
model.eval()

# ── SHAP with GradientExplainer ──
qm_p = model.encoder(batch["poly_z"].to(device), batch["poly_pos"].to(device),
                      batch["poly_edge"].to(device), batch["poly_batch"].to(device))
qm_s = model.encoder(batch["solv_z"].to(device), batch["solv_pos"].to(device),
                      batch["solv_edge"].to(device), batch["solv_batch"].to(device))
X = torch.cat([qm_p, qm_s, batch["temp"].unsqueeze(-1).to(device)], dim=-1).detach()

class QMHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = model.mlp
    def forward(self, x):
        return self.mlp(x)  # (batch, 1)

head = QMHead().to(device)
explainer = shap.GradientExplainer(head, X[:50])
shap_values = explainer.shap_values(X)
shap_values = np.array(shap_values).squeeze()

FEATURE_LABELS = [
    "μ (Polymer)", "α (Polymer)", "HOMO (Polymer)", "LUMO (Polymer)",
    "μ (Solvent)", "α (Solvent)", "HOMO (Solvent)", "LUMO (Solvent)",
    "Temperature",
]

shap_mean = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(shap_mean)

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(183/25.4, 80/25.4),
                                 gridspec_kw={"width_ratios": [0.4, 0.6]})

# Bar plot
ax1.barh(range(9), shap_mean[sorted_idx], color="#4c72b0", edgecolor="white", linewidth=0.3)
ax1.set_yticks(range(9))
ax1.set_yticklabels([FEATURE_LABELS[i] for i in sorted_idx], fontsize=6)
ax1.set_xlabel("Mean |SHAP|", fontsize=8)
ax1.set_title("A  QM feature importance", loc="left", fontsize=8, fontweight="bold")

# Beeswarm
for i in range(9):
    idx = sorted_idx[i]
    vals = shap_values[:, idx]
    x_vals = X[:, idx].cpu().numpy()
    # Normalize x for color
    x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min() + 1e-8)
    y_jitter = np.random.normal(i, 0.08, len(vals))
    ax2.scatter(vals, y_jitter, s=4, c=x_norm, cmap="RdYlBu_r", alpha=0.6)
ax2.axvline(0, color="gray", linewidth=0.5, linestyle="--")
ax2.set_yticks(range(9))
ax2.set_yticklabels([FEATURE_LABELS[i] for i in sorted_idx], fontsize=6)
ax2.set_xlabel("SHAP value", fontsize=8)
ax2.set_title("B  Impact on χ", loc="left", fontsize=8, fontweight="bold")

plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/qm_shap_{timestamp}.svg", bbox_inches="tight")
fig.savefig(f"{SAVE_DIR}/qm_shap_{timestamp}.png", dpi=300, bbox_inches="tight")
print(f"Saved: qm_shap_{timestamp}.svg / .png")

print("\nQM 属性重要性排名:")
for rank, idx in enumerate(sorted_idx[::-1], 1):
    print(f"  {rank}. {FEATURE_LABELS[idx]:20s}  |SHAP|={shap_mean[idx]:.4f}")
