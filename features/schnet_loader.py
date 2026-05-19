"""Polymer-solvent pair → dict with 3D structures for SchNet encoder"""
import pandas as pd, torch, numpy as np, re
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit.Chem import AllChem
from torch.utils.data import Dataset, DataLoader

CSV_PATH = "/Users/arcadio/flory_huggins/data/Nistane2022_polymer_solvent_chi.csv"
FEATURES_PKL = "/Users/arcadio/flory_huggins/features/rdkit_descriptors.pkl"
SCHNET_MAX_Z = 10


def parse_smiles(smi):
    """Try multiple strategies to cap polymer attachment points *"""
    import re
    # Strategy 1: replace [*] with [H] and bare * with C
    def try_replace(pattern, repl):
        return smi.replace(pattern, repl)

    candidates = []
    for repl in ["[H]", "[CH3]", ""]:
        s = smi.replace("[*]", repl).replace("*", "C")
        s = re.sub(r'\(\)', '', s)
        candidates.append(s)

    for s in candidates:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            return s
    return candidates[0]  # fallback to [H] variant


def get_3d_structure(smiles, seed=42):
    """SMILES → RDKit 3D conformer → z (atomic numbers), pos (coordinates)"""
    smi_clean = parse_smiles(smiles)
    mol = Chem.MolFromSmiles(smi_clean)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles.replace("*", ""))
        n = 1
        z = torch.zeros(n, dtype=torch.long)
        pos = torch.randn(n, 3, dtype=torch.float)
        return z, pos, None
    mol = Chem.RWMol(mol)
    try:
        succeeded = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if succeeded != 0:
            succeeded = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)
    except:
        succeeded = -1
    if succeeded != 0 or mol.GetNumConformers() == 0:
        n = mol.GetNumAtoms()
        z = torch.tensor([a.GetAtomicNum() if a.GetAtomicNum() <= SCHNET_MAX_Z else 0
                          for a in mol.GetAtoms()], dtype=torch.long)
        pos = torch.randn(n, 3, dtype=torch.float)
        return z, pos, None
    conf = mol.GetConformer()
    z = torch.tensor([a.GetAtomicNum() if a.GetAtomicNum() <= SCHNET_MAX_Z else 0
                      for a in mol.GetAtoms()], dtype=torch.long)
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
    return z, pos, mol


def build_edge_index(mol):
    """从 RDKit Mol 构建 edge_index (2, num_edges)"""
    if mol is None or mol.GetNumAtoms() < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    src, dst = [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [u, v]
        dst += [v, u]
    return torch.tensor([src, dst], dtype=torch.long)


class ChiGraphDataset(Dataset):
    """Precompute 3D structures + RDKit features for all polymer-solvent pairs"""
    def __init__(self, df, cache=True):
        original_idx = df.index.values  # save original index before reset
        self.df = df.reset_index(drop=True)
        # Load RDKit features (24-dim) — align by original DataFrame index
        feat_df = pd.read_pickle(FEATURES_PKL)
        rdkit_cols = [c for c in feat_df.columns if c.startswith(("Polymer_", "Solvent_"))]
        self.rdkit = torch.tensor(
            feat_df.loc[original_idx, rdkit_cols].values, dtype=torch.float)
        self._cache = None
        if cache:
            self._preprocess()

    def _preprocess(self):
        print(f"Preprocessing {len(self.df)} pairs...")
        data = []
        for i, row in self.df.iterrows():
            z_p, p_p, m_p = get_3d_structure(row["Polymer SMILES"])
            z_s, p_s, m_s = get_3d_structure(row["Solvent SMILES"])
            data.append({
                "poly_z": z_p, "poly_pos": p_p, "poly_edge": build_edge_index(m_p),
                "solv_z": z_s, "solv_pos": p_s, "solv_edge": build_edge_index(m_s),
                "rdkit": self.rdkit[len(data)],
                "temp": row["temperature"],
                "chi": row["chi"],
            })
        self._cache = data
        # Normalize temperature
        temps = torch.tensor([d["temp"] for d in data], dtype=torch.float)
        self.temp_mean, self.temp_std = temps.mean(), temps.std() + 1e-8
        for d in data:
            d["temp"] = (d["temp"] - self.temp_mean) / self.temp_std
        print(f"  Done. {len(data)} pairs cached.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self._cache is not None:
            d = self._cache[idx]
            return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
        # On-the-fly (slow, fallback)
        row = self.df.iloc[idx]
        z_p, p_p, m_p = get_3d_structure(row["Polymer SMILES"])
        z_s, p_s, m_s = get_3d_structure(row["Solvent SMILES"])
        return {
            "poly_z": z_p, "poly_pos": p_p, "poly_edge": build_edge_index(m_p),
            "solv_z": z_s, "solv_pos": p_s, "solv_edge": build_edge_index(m_s),
            "rdkit": self.rdkit[idx],
            "temp": row["temperature"],
            "chi": row["chi"],
        }


def collate_pairs(batch):
    """Custom collate for polymer-solvent pairs"""
    poly_z, poly_pos, poly_edge = [], [], []
    solv_z, solv_pos, solv_edge = [], [], []
    rdkit, temps, chis = [], [], []
    poly_b, solv_b = [], []
    npoly, nsolv = 0, 0

    for i, s in enumerate(batch):
        poly_z.append(s["poly_z"]); poly_pos.append(s["poly_pos"])
        e = s["poly_edge"] + npoly; poly_edge.append(e)
        npoly += len(s["poly_z"]); poly_b.extend([i] * len(s["poly_z"]))
        solv_z.append(s["solv_z"]); solv_pos.append(s["solv_pos"])
        e = s["solv_edge"] + nsolv; solv_edge.append(e)
        nsolv += len(s["solv_z"]); solv_b.extend([i] * len(s["solv_z"]))
        rdkit.append(s["rdkit"]); temps.append(s["temp"]); chis.append(s["chi"])

    return {
        "poly_z": torch.cat(poly_z), "poly_pos": torch.cat(poly_pos),
        "poly_edge": torch.cat(poly_edge, dim=1),
        "poly_batch": torch.tensor(poly_b),
        "solv_z": torch.cat(solv_z), "solv_pos": torch.cat(solv_pos),
        "solv_edge": torch.cat(solv_edge, dim=1),
        "solv_batch": torch.tensor(solv_b),
        "rdkit": torch.stack(rdkit),
        "temp": torch.tensor(temps, dtype=torch.float),
        "chi": torch.tensor(chis, dtype=torch.float),
    }


def get_dataloaders(batch_size=16, test_size=0.2, random_state=42):
    """Returns (train_loader, val_loader, test_loader)"""
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    from sklearn.model_selection import train_test_split
    pool_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    pool_df = pool_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n_val = int(len(pool_df) * 0.2)
    train_df, val_df = pool_df[:-n_val], pool_df[-n_val:]

    train_ds = ChiGraphDataset(train_df)
    val_ds = ChiGraphDataset(val_df)
    test_ds = ChiGraphDataset(test_df)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pairs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pairs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pairs),
    )


if __name__ == "__main__":
    ds, train_loader, val_loader = get_dataloader("pool", batch_size=4)
    batch = next(iter(train_loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:15s} shape={list(v.shape)} dtype={v.dtype}")
        else:
            print(f"{k:15s} = {v}")
