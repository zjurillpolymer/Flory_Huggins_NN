"""迁移能力测试：在 data_Chi 和 data_COSMO 上评估预训练模型"""
import sys, torch, numpy as np, pandas as pd
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, "/Users/arcadio/flory_huggins")

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import mean_absolute_error, r2_score
from models.attention_net import ChiPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DESC_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "RingCount",
    "MolMR", "BertzCT", "Chi1v", "Chi0v"
]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)


def calc_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [float("nan")] * len(DESC_NAMES)
    return list(calculator.CalcDescriptors(mol))


def load_and_evaluate(csv_path, model, label, split_underscore=True):
    """加载 csv，计算特征，评估模型"""
    df = pd.read_csv(csv_path)

    if split_underscore:
        # data_Chi/COSMO 格式：ps_pair = polymer_SMILES + "_" + solvent_SMILES
        pairs = df["ps_pair"].str.split("_", n=1)
        poly_smiles = pairs.str[0]
        solv_smiles = pairs.str[1]
    else:
        poly_smiles = df["Polymer SMILES"]
        solv_smiles = df["Solvent SMILES"]

    print(f"\n--- {label} ---")
    print(f"样本数: {len(df)}")

    # 计算描述符
    poly_feats = np.array([calc_descriptors(s) for s in poly_smiles])
    solv_feats = np.array([calc_descriptors(s) for s in solv_smiles])

    # 剔除无效 SMILES
    valid = ~(np.isnan(poly_feats).any(axis=1) | np.isnan(solv_feats).any(axis=1))
    X = np.concatenate([poly_feats[valid], solv_feats[valid]], axis=1)
    y = df["chi"].values[valid]
    print(f"有效: {valid.sum()}/{len(df)}")

    # 推理
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    print(f"MAE: {mae:.4f} | R²: {r2:.4f}")
    return mae, r2, y, pred


def main():
    # 加载模型（用 24-dim 的基线模型）
    model = ChiPredictor(in_channels=24).to(device)
    model.load_state_dict(torch.load(
        "/Users/arcadio/flory_huggins/models/best_model.pt", map_location=device))

    # Nistane 测试集（源域，参考基准）
    print("=" * 45)
    print("源域: Nistane 测试集(参考)")
    print("=" * 45)
    from features.data_loader import load_data as ld
    nist = ld(batch_size=256)
    desc_cols = [c for c in nist["test_df"].columns if c.startswith(("Polymer_", "Solvent_"))]
    X_nist = torch.tensor(nist["test_df"][desc_cols].values, dtype=torch.float32)
    y_nist = torch.tensor(nist["test_df"]["chi"].values, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(X_nist.to(device)).cpu().numpy()
    print(f"MAE: {mean_absolute_error(y_nist.numpy(), pred):.4f} | R²: {r2_score(y_nist.numpy(), pred):.4f}")

    # 迁移到 data_Chi
    load_and_evaluate("/Users/arcadio/flory_huggins/data/data_Chi.csv", model,
                       "目标域: data_Chi (实验 χ)")

    # 迁移到 data_COSMO
    load_and_evaluate("/Users/arcadio/flory_huggins/data/data_COSMO.csv", model,
                       "目标域: data_COSMO (COSMO-RS χ)")


if __name__ == "__main__":
    main()
