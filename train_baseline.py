import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from models.attention_net import encoder, MultiScaleAttention, Decoder, pool_loader, test_loader


class ChiPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.attention = MultiScaleAttention()
        self.decoder = Decoder()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = self.decoder(x)
        return x.squeeze(-1)


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
    model = ChiPredictor().to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    epochs = 100

    # validation split from pool (20% of pool for validation)
    all_X = []
    all_y = []
    for X, y in pool_loader:
        all_X.append(X)
        all_y.append(y)
    all_X = torch.cat(all_X)
    all_y = torch.cat(all_y)
    n_val = int(len(all_X) * 0.2)
    # use last n_val samples as val (already shuffled=False → deterministic)
    train_X, val_X = all_X[:-n_val], all_X[-n_val:]
    train_y, val_y = all_y[:-n_val], all_y[-n_val:]

    from torch.utils.data import TensorDataset, DataLoader
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=8, shuffle=False)

    best_mae = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae, val_r2 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "/Users/arcadio/flory_huggins/models/best_model.pt")

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val MAE: {val_mae:.4f} | val R²: {val_r2:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}")

    # final test evaluation
    model.load_state_dict(torch.load("/Users/arcadio/flory_huggins/models/best_model.pt"))
    test_loss, test_mae, test_r2 = evaluate(model, test_loader, criterion, device)
    print(f"\n=== Test set ===")
    print(f"MAE: {test_mae:.4f} | R²: {test_r2:.4f}")


if __name__ == "__main__":
    main()
