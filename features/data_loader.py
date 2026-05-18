"""数据加载：从 pkl 读取特征，划分 test/pool，生成 DataLoader"""

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

FEATURES_PATH = '/Users/arcadio/flory_huggins/features/rdkit_descriptors.pkl'
FEATURE_COLS_PREFIX = ("Polymer_", "Solvent_")
TARGET_COL = "chi"


def load_data(test_size=0.2, batch_size=8, random_state=42):
    dataset = pd.read_pickle(FEATURES_PATH)

    pool_df, test_df = train_test_split(dataset, test_size=test_size, random_state=random_state)

    feature_cols = [col for col in pool_df.columns if col.startswith(FEATURE_COLS_PREFIX)]

    def _to_loader(df, shuffle=False):
        X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        y = torch.tensor(df[TARGET_COL].values, dtype=torch.float32)
        return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)

    return {
        "pool_df": pool_df,
        "test_df": test_df,
        "pool_loader": _to_loader(pool_df, shuffle=False),
        "test_loader": _to_loader(test_df, shuffle=False),
        "feature_cols": feature_cols,
    }


def df_to_loader(df, batch_size=8, shuffle=False, feature_cols=None):
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col.startswith(FEATURE_COLS_PREFIX)]
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df[TARGET_COL].values, dtype=torch.float32)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
