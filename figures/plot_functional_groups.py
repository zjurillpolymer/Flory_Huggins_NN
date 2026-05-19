"""Functional group analysis (cf. Fig 5.4b, 5.5a)"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "/Users/arcadio/flory_huggins")

from rdkit import Chem
from rdkit.Chem import Fragments
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

# ── Load data ──
df = pd.read_csv("/Users/arcadio/flory_huggins/data/Nistane2022_polymer_solvent_chi.csv",
                  encoding="utf-8-sig")

# ── Functional group detection ──
# Get all fr_* functions from rdkit.Chem.Fragments
fg_funcs = [(name, getattr(Fragments, name))
            for name in dir(Fragments) if name.startswith("fr_")]
fg_funcs = sorted(fg_funcs, key=lambda x: x[0])

def get_fg_counts(smiles_list):
    """Return DataFrame: len(smiles) × len(fg_funcs)"""
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([0] * len(fg_funcs))
        else:
            rows.append([func(mol) for _, func in fg_funcs])
    return pd.DataFrame(rows, columns=[name for name, _ in fg_funcs])

print("Computing functional groups for Polymer SMILES...")
fg_poly = get_fg_counts(df["Polymer SMILES"].tolist())
print("Computing functional groups for Solvent SMILES...")
fg_solv = get_fg_counts(df["Solvent SMILES"].tolist())

# Combine both (mol1 + mol2 presence)
fg_both = ((fg_poly > 0) | (fg_solv > 0)).astype(int)

# ── Filter: keep functional groups present in at least 5% of samples ──
min_count = int(0.05 * len(df))
fg_mask = fg_both.sum() >= min_count
fg_filtered = fg_both.loc[:, fg_mask]
fg_names = fg_filtered.columns.tolist()

# ── Frequency (Fig 5.4b-like) ──
freq_poly = (fg_poly[fg_names] > 0).mean()
freq_solv = (fg_solv[fg_names] > 0).mean()

# Sort by overall frequency
order = (freq_poly + freq_solv).sort_values(ascending=False).index[:15]

# ── Correlation with χ (Fig 5.5a-like) ──
chi = df["chi"].values
corrs = []
for name in order:
    corr_poly = np.corrcoef((fg_poly[name] > 0).astype(float), chi)[0, 1]
    corr_solv = np.corrcoef((fg_solv[name] > 0).astype(float), chi)[0, 1]
    corrs.append((corr_poly, corr_solv))
corr_poly_vals, corr_solv_vals = zip(*corrs)

# ── Figure: 2 panels ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(183/25.4, 90/25.4),
                                 gridspec_kw={"width_ratios": [0.5, 0.5]})

# Panel A: Frequency bar chart
x = np.arange(len(order))
width = 0.35
bars1 = ax1.bar(x - width/2, freq_poly[order].values * 100, width,
                 label="Polymer", color="#4c72b0", edgecolor="white", linewidth=0.3)
bars2 = ax1.bar(x + width/2, freq_solv[order].values * 100, width,
                 label="Solvent", color="#c44e52", edgecolor="white", linewidth=0.3)
ax1.set_xticks(x)
# Clean up names: remove fr_ prefix, replace _ with space
short_names = [n.replace("fr_", "").replace("_", " ") for n in order]
ax1.set_xticklabels(short_names, rotation=45, ha="right", fontsize=5.5)
ax1.set_ylabel("Frequency (%)", fontsize=8)
ax1.legend(fontsize=7)
ax1.set_title("A  Functional group frequency", loc="left", fontsize=8, fontweight="bold")

# Panel B: Correlation with χ
x2 = np.arange(len(order))
ax2.scatter(range(len(order)), corr_poly_vals, s=20, c="#4c72b0", label="Polymer", zorder=3)
ax2.scatter(range(len(order)), corr_solv_vals, s=20, marker="s", c="#c44e52", label="Solvent", zorder=3)
for i in range(len(order)):
    ax2.plot([i, i], [corr_poly_vals[i], corr_solv_vals[i]], color="gray", linewidth=0.5, alpha=0.5)
ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax2.set_xticks(range(len(order)))
ax2.set_xticklabels(short_names, rotation=45, ha="right", fontsize=5.5)
ax2.set_ylabel("Correlation with χ", fontsize=8)
ax2.legend(fontsize=7, loc="lower right")
ax2.set_title("B  FG-χ correlation", loc="left", fontsize=8, fontweight="bold")

plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/functional_groups_{timestamp}.svg", bbox_inches="tight")
fig.savefig(f"{SAVE_DIR}/functional_groups_{timestamp}.png", dpi=300, bbox_inches="tight")
print(f"Saved: functional_groups_{timestamp}.svg / .png")

# Print top correlations
print("\nTop +correlations (Polymer):")
for name, c in sorted(zip(order, corr_poly_vals), key=lambda x: -abs(x[1]))[:5]:
    print(f"  {name:25s}  r = {c:+.3f}")
print("\nTop +correlations (Solvent):")
for name, c in sorted(zip(order, corr_solv_vals), key=lambda x: -abs(x[1]))[:5]:
    print(f"  {name:25s}  r = {c:+.3f}")
