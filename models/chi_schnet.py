"""ChiSchNet: RDKit descriptors + SchNet QM properties → χ"""
import torch, torch.nn as nn, sys
sys.path.insert(0, "/Users/arcadio/Slimnet")
sys.path.insert(0, "/Users/arcadio/Slimnet/base_model_molecule_encoder")
from Schnet_model_monomer import Schnet_monomer


class ChiSchNet(nn.Module):
    def __init__(self, freeze_encoder=True):
        super().__init__()
        self.encoder = Schnet_monomer(hidden_dim=128, n_layers=6)
        ckpt = torch.load("/Users/arcadio/flory_huggins/models/best_schnet.pt",
                           map_location="cpu")
        self.encoder.load_state_dict(ckpt)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("[ChiSchNet] Encoder frozen")
        else:
            print("[ChiSchNet] Encoder fine-tunable")

        # RDKit(24) + QM(8: 4×2) + temp(1) = 33 → χ
        self.mlp = nn.Sequential(
            nn.Linear(33, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, batch):
        # SchNet QM properties for polymer and solvent
        qm_poly = self.encoder(
            batch["poly_z"], batch["poly_pos"],
            batch["poly_edge"], batch["poly_batch"])  # (B, 4)
        qm_solv = self.encoder(
            batch["solv_z"], batch["solv_pos"],
            batch["solv_edge"], batch["solv_batch"])  # (B, 4)

        # RDKit descriptors (already in batch)
        x = torch.cat([
            batch["rdkit"],        # (B, 24)
            qm_poly, qm_solv,      # (B, 8)
            batch["temp"].unsqueeze(-1),  # (B, 1)
        ], dim=-1)  # (B, 33)

        return self.mlp(x).squeeze(-1)
