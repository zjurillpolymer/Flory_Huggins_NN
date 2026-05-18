import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score
from models.attention_net import ChiPredictor
from features.data_loader import load_data, df_to_loader


def train_one_epoch(model, loader, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    preds, targets = [], []
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += criterion(pred, y).item()
            preds.append(pred.cpu())
            targets.append(y.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    return total_loss / len(loader), mae, r2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_data(batch_size=8)
    pool_df = data["pool_df"]
    test_loader = data["test_loader"]

    pool_df = pool_df.sample(frac=1, random_state=42)  # 打乱
    n_val = int(len(pool_df) * 0.2)
    train_loader = df_to_loader(pool_df[:-n_val], batch_size=8, shuffle=True)
    val_loader = df_to_loader(pool_df[-n_val:], batch_size=8, shuffle=False)

    model = ChiPredictor().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_mae = float("inf")
    for epoch in range(1, 101):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae, val_r2 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "/Users/arcadio/flory_huggins/models/best_model.pt")

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val MAE: {val_mae:.4f} | val R²: {val_r2:.4f}")

    model.load_state_dict(torch.load("/Users/arcadio/flory_huggins/models/best_model.pt"))
    test_loss, test_mae, test_r2 = evaluate(model, test_loader, criterion, device)
    print(f"\n=== Test set ===")
    print(f"MAE: {test_mae:.4f} | R²: {test_r2:.4f}")


if __name__ == "__main__":
    main()
