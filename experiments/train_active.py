"""主动学习主循环：MC Dropout / Ensemble 不确定性采样 vs 随机采样"""

import sys, os, torch, torch.nn as nn, numpy as np, pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, "/Users/arcadio/flory_huggins")
from models.attention_net import ChiPredictor
from features.data_loader import load_data, df_to_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(train_df, val_df, epochs=60, lr=1e-3):
    """用给定标注集训练模型，返回训练好的模型"""
    train_loader = df_to_loader(train_df, batch_size=8, shuffle=True)
    val_loader = df_to_loader(val_df, batch_size=8, shuffle=False)

    model = ChiPredictor().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_mae = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 验证
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds.append(model(X).cpu())
                targets.append(y.cpu())
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        mae = mean_absolute_error(targets, preds)
        scheduler.step(mae)

        if mae < best_mae:
            best_mae = mae
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model


def mc_dropout_uncertainty(model, loader, num_samples=30):
    """MC Dropout 不确定性估计，返回每个样本的方差"""
    model.train()
    all_preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            batch_preds = []
            for X, _ in loader:
                X = X.to(device)
                batch_preds.append(model(X).cpu().numpy())
            all_preds.append(np.concatenate(batch_preds))
    return np.var(np.array(all_preds), axis=0)


def ensemble_uncertainty(labeled_df, val_df, unlabeled_df, num_models=4, epochs=40):
    """训练 num_models 个模型，用预测方差作为不确定性"""
    models = []
    for i in range(num_models):
        model = train_model(labeled_df, val_df, epochs=epochs)
        models.append(model)

    # 对所有模型做推理，算方差
    ul_loader = df_to_loader(unlabeled_df, batch_size=32, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for model in models:
            model.eval()
            batch_preds = []
            for X, _ in ul_loader:
                X = X.to(device)
                batch_preds.append(model(X).cpu().numpy())
            all_preds.append(np.concatenate(batch_preds))
    return np.var(np.array(all_preds), axis=0)


def evaluate(model, loader):
    """在 DataLoader 上评估 MAE 和 R²"""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds.append(model(X).cpu())
            targets.append(y.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return mean_absolute_error(targets, preds), r2_score(targets, preds)


def run_active_learning(pool_df, test_loader, strategy="active", initial_size=100,
                        n_rounds=40, samples_per_round=3, uncertainty_method="mc"):
    """
    strategy: "active" → 不确定性采样, "random" → 随机采样
    uncertainty_method: "mc" → MC Dropout, "ensemble" → Ensemble
    """
    # 初始划分
    labeled = pool_df.sample(initial_size, random_state=42)
    unlabeled = pool_df.drop(labeled.index).reset_index(drop=True)
    # 从 labeled 中再切 20% 做验证
    labeled = labeled.sample(frac=1, random_state=42)
    n_val = int(len(labeled) * 0.2)
    val_df = labeled.iloc[:n_val]
    train_labeled = labeled.iloc[n_val:]

    history = {"round": [], "n_labeled": [], "mae": [], "r2": []}
    current_labeled = train_labeled.copy()

    for rnd in range(n_rounds):
        # 训练
        model = train_model(current_labeled, val_df, epochs=80)

        # 在 test set 上评估当前精度
        mae, r2 = evaluate(model, test_loader)
        history["round"].append(rnd + 1)
        history["n_labeled"].append(len(current_labeled))
        history["mae"].append(mae)
        history["r2"].append(r2)

        if (rnd + 1) % 10 == 0:
            flag = "✓" if strategy == "active" else "○"
            print(f"{flag} Round {rnd+1:3d} | labeled: {len(current_labeled):4d} | MAE: {mae:.4f} | R²: {r2:.4f}")

        if len(unlabeled) < samples_per_round:
            break

        if strategy == "active":
            if uncertainty_method == "ensemble":
                uncert = ensemble_uncertainty(current_labeled, val_df, unlabeled, epochs=40)
            else:
                unlabeled_loader = df_to_loader(unlabeled, batch_size=32, shuffle=False)
                uncert = mc_dropout_uncertainty(model, unlabeled_loader)
            pick_idx = np.argsort(uncert)[-samples_per_round:]
        else:
            # 随机采样
            pick_idx = np.random.choice(len(unlabeled), samples_per_round, replace=False)

        # 模拟标注：移到标注集
        picked = unlabeled.iloc[pick_idx]
        current_labeled = pd.concat([current_labeled, picked], ignore_index=True)
        unlabeled = unlabeled.drop(unlabeled.index[pick_idx]).reset_index(drop=True)

    return pd.DataFrame(history)


def main():
    print("=" * 50)
    print("Active Learning for χ Prediction")
    print("=" * 50)

    data = load_data(batch_size=8)
    test_loader = data["test_loader"]
    pool_df = data["pool_df"].sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Pool: {len(pool_df)} 条 | Test: {len(test_loader.dataset)} 条\n")

    print("▶ Running Random Sampling (baseline) ...")
    random_history = run_active_learning(pool_df, test_loader, strategy="random")
    random_history.to_pickle("/Users/arcadio/flory_huggins/experiments/history_random.pkl")

    print("\n▶ Running Active Learning (MC Dropout) ...")
    mc_history = run_active_learning(pool_df, test_loader, strategy="active", uncertainty_method="mc")
    mc_history.to_pickle("/Users/arcadio/flory_huggins/experiments/history_mc.pkl")

    print("\n▶ Running Active Learning (Ensemble) ...")
    en_history = run_active_learning(pool_df, test_loader, strategy="active", uncertainty_method="ensemble",
                                      n_rounds=20, samples_per_round=3)
    en_history.to_pickle("/Users/arcadio/flory_huggins/experiments/history_ensemble.pkl")

    # 最终对比
    print("\n" + "=" * 55)
    print("Final Comparison")
    print("=" * 55)
    for name, hist in [("Random", random_history), ("MC Dropout", mc_history), ("Ensemble", en_history)]:
        f = hist.iloc[-1]
        print(f"  {name:10s}  MAE={f['mae']:.4f}  R²={f['r2']:.4f}")

    print("\n历史记录已保存")


if __name__ == "__main__":
    main()
