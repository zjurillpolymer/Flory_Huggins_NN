# Flory-Huggins χ Parameter Prediction with Active Learning

Reproduction of Chapter 5 from Xu Han's PhD thesis: *"主动学习方法预测嵌段共聚物 Flory-Huggins 参数"*.

## Project Structure

```
├── data/
│   ├── README.md                    # 数据集说明
│   ├── Nistane2022_polymer_solvent_chi.csv  # 聚合物-溶剂 χ 数据 (1586条)
│   ├── data_Chi.csv                 # MTL-Chi 实验 χ 数据 (1190条)
│   ├── data_COSMO.csv               # COSMO-RS 计算 χ 数据 (1206条)
│   └── data_PI.csv                  # 聚合物信息
│
├── features/
│   ├── data_loader.py               # 数据加载 (train/test/pool划分)
│   ├── rdkit_descriptors.py         # RDKit 12个分子描述符计算
│   └── rdkit_descriptors.pkl       # 计算结果 (24维特征)
│
├── models/
│   ├── attention_net.py             # 注意力网络 (Encoder + MultiHeadAttention + Decoder)
│   ├── uncertainty.py               # 不确定性量化 (MC Dropout + Ensemble)
│   └── best_model.pt                # 训练好的基线模型权重
│
├── experiments/
│   └── train_baseline.py            # 基线模型训练脚本
│
├── ROADMAP.md
└── README.md
```

## Model Architecture (Section 5.2.3)

```
Input (24-dim: 12 polymer + 12 solvent descriptors)
    ↓
Linear(24→64) + LayerNorm + ReLU + Dropout(0.1)    ← Encoder
    ↓
Multi-Head Self-Attention (dim=64, heads=8)         ← Attention
    ↓
Dropout(0.1)
    ↓
Linear(64→1)                                         ← Decoder/Prediction
    ↓
Output: χ (Flory-Huggins parameter)
```

## Uncertainty Quantification (Section 5.2.4)

Two methods implemented:

| Method | Description | Inference Cost |
|--------|-------------|----------------|
| **MC Dropout** | Single model, 50 forward passes with dropout enabled | ~1× |
| **Ensemble** | 4 independently trained models, 1 forward pass each | ~4× training |

Usage:
```bash
python -m models.uncertainty              # MC Dropout
python -m models.uncertainty ensemble     # Ensemble (train + inference)
```

### Benchmark Results

| Method | MAE | R² | Unc-Error Correlation |
|--------|-----|----|----------------------|
| Baseline (single model) | 0.244 | 0.643 | — |
| MC Dropout (50 samples) | **0.231** | **0.759** | 0.170 |
| Ensemble (4 models) | 0.236 | 0.746 | **0.279** |

## Training

```bash
python -m experiments.train_baseline
```

Hyperparameters (matching thesis):
- Optimizer: AdamW (lr=1e-3, weight_decay=0.01)
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Gradient clipping: 1.0
- Batch size: 8
- Dropout: 0.1
- Loss: MSE

## Data Note

The thesis uses block copolymer χ data (1221 points from CHiMaD/PPPdb) as the target task,
which is not publicly downloadable. This reproduction uses polymer-solvent χ data (1586 points from Nistane et al.)
as a substitute. The active learning methodology is dataset-agnostic.

## References

- Xu, H. (2025). 小数据集场景下的机器学习方法辅助聚合物材料设计与研究. 浙江大学博士论文.
- Nistane, D. et al. (2022). *Estimation of the Flory-Huggins interaction parameter...* MRS Advances.
- Maruyama, M. et al. (2023). *Multitask Machine Learning to Predict Polymer–Solvent Miscibility...* Macromolecules.
