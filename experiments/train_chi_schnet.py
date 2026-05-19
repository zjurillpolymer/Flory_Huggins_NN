"""Train ChiSchNet: SchNet predicts QM properties → MLP → χ"""
import torch, torch.nn as nn, numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score

import sys
sys.path.insert(0, "/Users/arcadio/flory_huggins")
from features.schnet_loader import get_dataloaders
from models.chi_schnet import ChiSchNet

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")


def train_one_epoch(model, loader, optimizer, criterion, clip=1.0):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()
        loss = criterion(model(batch), batch["chi"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    preds, targets = [], []
    total_loss = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pred = model(batch)
        total_loss += criterion(pred, batch["chi"]).item()
        preds.append(pred.cpu())
        targets.append(batch["chi"].cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return total_loss / len(loader), mean_absolute_error(targets, preds), r2_score(targets, preds)


def main():
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

    model = ChiSchNet(freeze_encoder=True).to(device)

    # 只训练 MLP head（冻结 SchNet）
    optimizer = AdamW(model.mlp.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_mae = float("inf")
    for epoch in range(1, 101):
        tl = train_one_epoch(model, train_loader, optimizer, criterion)
        vl, vm, vr = evaluate(model, val_loader, criterion)
        scheduler.step(vl)
        if vm < best_mae:
            best_mae = vm
            torch.save(model.state_dict(), "/Users/arcadio/flory_huggins/models/best_chischnet.pt")
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train loss: {tl:.4f} | val MAE: {vm:.4f} | val R²: {vr:.4f}")

    # Test
    model.load_state_dict(torch.load("/Users/arcadio/flory_huggins/models/best_chischnet.pt"))
    _, tm, tr = evaluate(model, test_loader, criterion)
    print("\n=== Test (RDKit + SchNet combined) ===")
    print(f"MAE: {tm:.4f} | R²: {tr:.4f}")
    print(f"(Baseline RDKit model: MAE=0.244, R²=0.643)")


if __name__ == "__main__":
    main()
