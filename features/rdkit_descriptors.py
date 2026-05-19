import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

DATA_PATH = "/Users/arcadio/flory_huggins/data/Nistane2022_polymer_solvent_chi.csv"
SAVE_PATH = "/Users/arcadio/flory_huggins/features/rdkit_descriptors.pkl"

# 论文 5.2.3.1 节的 12 个分子描述符
DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "RingCount",
    "MolMR", "BertzCT", "Chi1v", "Chi0v"
]

calculator = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTOR_NAMES)


def calc_descriptors(smi: str):
    """单条 SMILES → 12 维描述符向量"""
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return list(calculator.CalcDescriptors(mol))
    return [float("nan")] * len(DESCRIPTOR_NAMES)


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"原始数据: {len(df)} 条")

    # 计算 Polymer 和 Solvent 的描述符
    poly_df = pd.DataFrame(
        df["Polymer SMILES"].apply(calc_descriptors).tolist(),
        columns=[f"Polymer_{n}" for n in DESCRIPTOR_NAMES],
        index=df.index,
    )
    solv_df = pd.DataFrame(
        df["Solvent SMILES"].apply(calc_descriptors).tolist(),
        columns=[f"Solvent_{n}" for n in DESCRIPTOR_NAMES],
        index=df.index,
    )

    # 拼接：原始列 + Polymer(12) + Solvent(12) = 24 维特征
    result = pd.concat([df, poly_df, solv_df], axis=1)

    # ── 物理约束特征（Hansen 参数近似）──
    # 色散力: MolMR 差异平方
    result["Phys_Dispersion_sq"] = (poly_df["Polymer_MolMR"] - solv_df["Solvent_MolMR"]) ** 2
    # 极性: TPSA 差异平方
    result["Phys_Polarity_sq"] = (poly_df["Polymer_TPSA"] - solv_df["Solvent_TPSA"]) ** 2
    # 氢键: (HBA+HBD) 差异平方
    poly_HB = poly_df["Polymer_NumHAcceptors"] + poly_df["Polymer_NumHDonors"]
    solv_HB = solv_df["Solvent_NumHAcceptors"] + solv_df["Solvent_NumHDonors"]
    result["Phys_HBond_sq"] = (poly_HB - solv_HB) ** 2

    result.to_pickle(SAVE_PATH)

    print(f"\n✓ 保存至 features/rdkit_descriptors.pkl")
    print(f"  形状: {result.shape} ({len(df)} 条 × {len(result.columns)} 列)")
    print(f"  特征: Polymer(12) + Solvent(12) + Phys(3) = 27 维")

    valid = result.dropna(subset=[f"Polymer_{n}" for n in DESCRIPTOR_NAMES]
                                   + [f"Solvent_{n}" for n in DESCRIPTOR_NAMES])
    print(f"  有效样本: {len(valid)}/{len(df)}")


if __name__ == "__main__":
    main()