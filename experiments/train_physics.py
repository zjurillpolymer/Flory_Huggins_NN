"""物理约束作为 Loss 正则项"""
import torch, torch.nn as nn, numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score
from models.attention_net import ChiPredictor
from features.data_loader import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PHYS_LAMBDA = 0.1


# 用全量 pool 的统计做标准化，保证一致性
_PHYS_STATS = None

def compute_chi_phys(df):
    """从原始特征计算 Hansen 近似 χ"""
    global _PHYS_STATS
    disp = (df["Polymer_MolMR"] - df["Solvent_MolMR"]) ** 2
    pol = (df["Polymer_TPSA"] - df["Solvent_TPSA"]) ** 2
    hb_poly = df["Polymer_NumHAcceptors"] + df["Polymer_NumHDonors"]
    hb_solv = df["Solvent_NumHAcceptors"] + df["Solvent_NumHDonors"]
    hb = (hb_poly - hb_solv) ** 2
    raw = (disp + pol + hb).values.astype(np.float32)
    if _PHYS_STATS is None:
        chi_mean, chi_std = df["chi"].mean(), df["chi"].std()
        _PHYS_STATS = (chi_mean, chi_std, raw.mean(), raw.std())
    chi_mean, chi_std, r_mean, r_std = _PHYS_STATS
    return (raw - r_mean) / r_std * chi_std + chi_mean


def make_phys_loader(df, batch_size=8, shuffle=False, feature_cols=None):
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith(("Polymer_", "Solvent_"))]
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df["chi"].values, dtype=torch.float32)
    chi_phys = torch.tensor(compute_chi_phys(df), dtype=torch.float32)
    return DataLoader(TensorDataset(X, y, chi_phys), batch_size=batch_size, shuffle=shuffle)


def train_phys(model, loader, optimizer, mse_loss, lambda_reg=PHYS_LAMBDA, clip=1.0):
    model.train()
    total_loss = 0
    for X, y, chi_p in loader:
        X, y, chi_p = X.to(device), y.to(device), chi_p.to(device)
        optimizer.zero_grad()
        pred = model(X)
        mse = mse_loss(pred, y)
        phys = mse_loss(pred, chi_p)
        loss = mse + lambda_reg * phys
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, mse_loss):
    model.eval()
    preds, targets = [], []
    total_loss = 0
    with torch.no_grad():
        for X, y, _ in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += mse_loss(pred, y).item()
            preds.append(pred.cpu())
            targets.append(y.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return total_loss / len(loader), mean_absolute_error(targets, preds), r2_score(targets, preds)


def main():
    data = load_data(batch_size=8)
    pool_df = data["pool_df"].sample(frac=1, random_state=42)
    test_loader = make_phys_loader(data["test_df"], batch_size=8)

    n_val = int(len(pool_df) * 0.2)
    train_loader = make_phys_loader(pool_df[:-n_val], batch_size=8, shuffle=True)
    val_loader = make_phys_loader(pool_df[-n_val:])

    model = ChiPredictor(in_channels=24).to(device)
    print(f"Model: 24-dim + physics regularization (λ={PHYS_LAMBDA})")

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    mse_loss = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_mae = float("inf")
    for epoch in range(1, 201):
        tl = train_phys(model, train_loader, optimizer, mse_loss)
        vl, vm, vr = evaluate(model, val_loader, mse_loss)
        scheduler.step(vl)
        if vm < best_mae:
            best_mae = vm
            torch.save(model.state_dict(), "/Users/arcadio/flory_huggins/models/best_physics.pt")
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train loss: {tl:.4f} | val MAE: {vm:.4f} | val R²: {vr:.4f}")

    model.load_state_dict(torch.load("/Users/arcadio/flory_huggins/models/best_physics.pt"))
    _, tm, tr = evaluate(model, test_loader, mse_loss)
    print(f"\n=== Test (physics regularization, λ={PHYS_LAMBDA}) ===")
    print(f"MAE: {tm:.4f} | R²: {tr:.4f}")
    print(f"(Baseline: MAE=0.244, R²=0.643)")


if __name__ == "__main__":
    main()
