"""训练含温度特征的模型 + 单调递减约束 (∂χ/∂T ≤ 0)"""
import torch, torch.nn as nn, numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from models.attention_net import ChiPredictor
from features.data_loader import load_data
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MONO_LAMBDA = 0.5     # 单调约束权重
TEMP_DELTA = 0.1      # 温度扰动幅度（归一化后约 5-6K）


def make_loader(df, feature_cols, batch_size=8, shuffle=False):
    X = df[feature_cols].values.astype(np.float32)
    y = df["chi"].values.astype(np.float32)
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                      batch_size=batch_size, shuffle=shuffle)


def monotonicity_loss(model, X, temp_idx):
    """强制 ∂χ/∂T ≤ 0：χ(T+δ) < χ(T) 应成立"""
    X_plus = X.clone()
    X_plus[:, temp_idx] += TEMP_DELTA
    chi_T = model(X)
    chi_T_plus = model(X_plus)
    # 如果 χ(T+δ) > χ(T)，则 penalty = (diff/δ)²
    diff = chi_T_plus - chi_T  # 应为负值
    penalty = torch.relu(diff) ** 2
    return penalty.mean()


def main():
    data = load_data()
    pool_df = data["pool_df"].sample(frac=1, random_state=42)
    test_df = data["test_df"]

    desc_cols = [c for c in pool_df.columns if c.startswith(("Polymer_", "Solvent_"))]
    n_val = int(len(pool_df) * 0.2)

    train_df = pool_df[:-n_val].copy()
    val_df = pool_df[-n_val:].copy()

    # 温度标准化
    temp_scaler = StandardScaler()
    train_df["temp_norm"] = temp_scaler.fit_transform(train_df[["temperature"]])
    val_df["temp_norm"] = temp_scaler.transform(val_df[["temperature"]])
    test_df = test_df.copy()
    test_df["temp_norm"] = temp_scaler.transform(test_df[["temperature"]])

    feature_cols = desc_cols + ["temp_norm"]
    n_features = len(feature_cols)
    temp_idx = n_features - 1  # 温度在最后一列

    train_loader = make_loader(train_df, feature_cols, batch_size=8, shuffle=True)
    val_loader = make_loader(val_df, feature_cols)
    test_loader = make_loader(test_df, feature_cols)

    model = ChiPredictor(in_channels=n_features).to(device)
    print(f"特征维度: {n_features} | 单调约束 λ={MONO_LAMBDA}")

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_mae = float("inf")
    for epoch in range(1, 101):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            mse = criterion(model(X), y)
            mono = monotonicity_loss(model, X, temp_idx)
            loss = mse + MONO_LAMBDA * mono
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        preds, targets = [], []
        vio_count = 0
        total_count = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds.append(model(X).cpu())
                targets.append(y.cpu())
                # 单调性检查
                X_plus = X.clone()
                X_plus[:, temp_idx] += TEMP_DELTA
                vio_count += (model(X_plus) > model(X)).sum().item()
                total_count += X.size(0)
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        vm = mean_absolute_error(targets, preds)
        vr = r2_score(targets, preds)
        vio_pct = vio_count / total_count * 100
        scheduler.step(vm)
        if vm < best_mae:
            best_mae = vm
            torch.save(model.state_dict(), "/Users/arcadio/flory_huggins/models/best_with_temp.pt")
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | val MAE: {vm:.4f} | val R²: {vr:.4f} | 违逆: {vio_pct:.1f}%")

    model.load_state_dict(torch.load("/Users/arcadio/flory_huggins/models/best_with_temp.pt"))
    model.eval()
    preds, targets = [], []
    vio_test = 0
    n_test = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds.append(model(X).cpu())
            targets.append(y.cpu())
            X_plus = X.clone()
            X_plus[:, temp_idx] += TEMP_DELTA
            vio_test += (model(X_plus) > model(X)).sum().item()
            n_test += X.size(0)
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    tm = mean_absolute_error(targets, preds)
    tr = r2_score(targets, preds)
    print(f"\n=== Test (24 RDKit + 归一化温度 + 单调约束) ===")
    print(f"MAE: {tm:.4f} | R²: {tr:.4f}")
    print(f"单调违逆率: {vio_test/n_test*100:.1f}%")

    # ── 温度-χ 关系验证 ──
    print("\n=== 温度-χ 关系验证 ===")
    X_sample, y_sample = next(iter(test_loader))
    temps_raw = [273, 323, 373, 423, 473, 523]
    temps_norm = temp_scaler.transform(np.array(temps_raw).reshape(-1, 1)).flatten()

    for i in range(5):
        preds_at_temps = []
        for t_norm in temps_norm:
            X_mod = X_sample.clone()
            X_mod[i, -1] = t_norm  # 只改第 i 个样本的温度
            p = model(X_mod[i:i+1].to(device)).item()
            preds_at_temps.append(p)
        trend = "✓ ↓" if preds_at_temps[0] > preds_at_temps[-1] else "✗ ↑"
        print(f"  样本 {i+1}: χ(273K)={preds_at_temps[0]:.3f} → χ(523K)={preds_at_temps[-1]:.3f}  {trend}")

    # 统计测试集上温度-预测的 Pearson 相关（每个样本不同温度）
    print("\n测试集整体温度-χ 相关性:")
    all_temps = temp_scaler.transform(test_df[["temperature"]]).flatten()
    all_chi_pred = preds
    corr = np.corrcoef(all_temps, all_chi_pred)[0, 1]
    print(f"  r(pred_temp, pred_χ) = {corr:.3f}  (期望: 负值)")


if __name__ == "__main__":
    main()
