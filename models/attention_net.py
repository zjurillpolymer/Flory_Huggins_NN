"""论文 5.2.3 节：基于注意力机制的 χ 参数预测模型"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """特征编码器：Linear(24→64) + LayerNorm + ReLU"""
    def __init__(self, in_channels=24, out_channels=64):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.layernorm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.layernorm(x)
        x = self.relu(x)
        return x


class MultiScaleAttention(nn.Module):
    """多尺度自注意力层"""
    def __init__(self, dim=64, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(1)           # (batch, 64) → (batch, 1, 64)
        out, _ = self.attn(x, x, x)  # Q=K=V 自注意力
        return out.squeeze(1)        # (batch, 1, 64) → (batch, 64)


class Decoder(nn.Module):
    """预测层：Linear(64→1)"""
    def __init__(self, in_channels=64):
        super().__init__()
        self.linear = nn.Linear(in_channels, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class ChiPredictor(nn.Module):
    """完整模型"""
    def __init__(self, in_channels=24, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.dropout = nn.Dropout(dropout)
        self.attention = MultiScaleAttention()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = self.decoder(x)
        return x
