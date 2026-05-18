import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
from torch import nn



'''读取数据集，划分test/pool两个集合和batch，特征编码器'''

features_path='/Users/arcadio/flory_huggins/features/rdkit_descriptors.pkl'

dataset=pd.read_pickle(features_path)

# print(dataset.columns)
'''
Index(['Polymer', 'Polymer SMILES', 'Solvent', 'Solvent SMILES', 'chi',
       'temperature', 'Reference Source', 'Polymer_MolWt', 'Polymer_MolLogP',
       'Polymer_TPSA', 'Polymer_NumHDonors', 'Polymer_NumHAcceptors',
       'Polymer_NumRotatableBonds', 'Polymer_NumAromaticRings',
       'Polymer_RingCount', 'Polymer_MolMR', 'Polymer_BertzCT',
       'Polymer_Chi1v', 'Polymer_Chi0v', 'Solvent_MolWt', 'Solvent_MolLogP',
       'Solvent_TPSA', 'Solvent_NumHDonors', 'Solvent_NumHAcceptors',
       'Solvent_NumRotatableBonds', 'Solvent_NumAromaticRings',
       'Solvent_RingCount', 'Solvent_MolMR', 'Solvent_BertzCT',
       'Solvent_Chi1v', 'Solvent_Chi0v'],
      dtype='object')
'''

pool_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

FEATURE_COLS = [col for col in pool_dataset.columns if col.startswith(("Polymer_", "Solvent_"))]
TARGET_COL = "chi"

def make_loader(df, batch_size=32, shuffle=False):
    X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
    y = torch.tensor(df[TARGET_COL].values, dtype=torch.float32)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)

pool_loader = make_loader(pool_dataset, shuffle=False)
test_loader = make_loader(test_dataset, shuffle=False)



class encoder(nn.Module):
    def __init__(self,in_channels=24,out_channels=64):
        super().__init__()
        self.linear=torch.nn.Linear(in_channels,out_channels)
        self.layernorm=torch.nn.LayerNorm(normalized_shape=out_channels, eps=1e-5, elementwise_affine=True)
        self.relu=torch.nn.ReLU()

    def forward(self,x):
        x=self.linear(x)
        x=self.layernorm(x)
        x=self.relu(x)
        return x



'''Attention'''


class MultiScaleAttention(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)

    def forward(self, x):
        # x: (batch, 64) → (batch, 1, 64)
        x = x.unsqueeze(1)
        # Q=K=V 都来自 x，标准的 self-attention
        out, _ = self.attn(x, x, x)
        return out.squeeze(1)




'''decoder'''
class Decoder(nn.Module):
    def __init__(self, feature_dim=64,out_channels=1):
        super().__init__()
        self.linear=torch.nn.Linear(feature_dim, out_channels)

    def forward(self,x):
        x = self.linear(x)
        return x




'''train'''
