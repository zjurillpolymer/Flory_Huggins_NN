# Flory-Huggins χ Parameter Prediction with Active Learning

Reproduction of Chapter 5 from Xu Han's PhD thesis: *"主动学习方法预测嵌段共聚物 Flory-Huggins 参数"*.

## Project Structure

```
├── data/
│   ├── README.md                             # 数据集说明
│   ├── Nistane2022_polymer_solvent_chi.csv   # 聚合物-溶剂 χ 数据 (1586条)
│   ├── data_Chi.csv                          # MTL-Chi 实验 χ 数据 (1190条)
│   ├── data_COSMO.csv                        # COSMO-RS 计算 χ 数据 (1206条)
│   └── data_PI.csv                           # 聚合物信息
│
├── features/
│   ├── data_loader.py                        # 数据加载 (train/test/pool划分)
│   ├── rdkit_descriptors.py                  # RDKit 12个分子描述符计算 + 物理约束特征
│   └── rdkit_descriptors.pkl                 # 计算结果 (27维: 24描述符 + 3物理约束)
│
├── models/
│   ├── attention_net.py                      # 注意力网络 (Encoder + MultiHeadAttention + Decoder)
│   ├── uncertainty.py                        # 不确定性量化 (MC Dropout + Ensemble)
│   ├── best_model.pt                         # 基线模型权重 (MAE=0.244)
│   └── best_physics.pt                       # 物理约束模型权重 (MAE=0.237)
│
├── experiments/
│   ├── train_baseline.py                     # 基线模型训练
│   ├── train_active.py                       # 主动学习主循环 (MC Dropout / Ensemble vs Random)
│   └── train_physics.py                      # 物理正则化 Loss 训练
│
├── figures/
│   ├── summary_*.png/svg                     # 不确定性量化结果
│   ├── active_learning_*.png/svg             # 主动学习对比曲线
│   ├── interpret_*.png/svg                   # SHAP 特征重要性 + 相关热图
│   ├── shap_dependence_*.png/svg             # SHAP 依赖图 (Top 6特征)
│   └── functional_groups_*.png/svg          # 官能团频率 + FG-χ 相关性
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
Multi-Head Self-Attention (dim=64, heads=8)         ← MultiScaleAttention
    ↓
Dropout(0.1)
    ↓
Linear(64→1)                                         ← Decoder / Prediction
    ↓
Output: χ (Flory-Huggins parameter)
```

## Model Performance

| Model | MAE | R² | Note |
|-------|-----|----|------|
| Baseline | 0.244 | 0.643 | 24-dim, 100 epochs |
| MC Dropout (T=50) | **0.231** | **0.759** | Uncertainty by dropout variance |
| Ensemble (4 models) | 0.236 | 0.746 | Uncertainty by model disagreement |
| Physics regularization | 0.237 | 0.639 | Hansen-approximated loss term (λ=0.1) |

## Interpretability (Section 5.3.5)

### SHAP Feature Importance (Top 8)

| Rank | Feature | Mean \|SHAP\| | Physical meaning |
|------|---------|-------------|------------------|
| 1 | **BertzCT (Polymer)** | 0.317 | Molecular complexity |
| 2 | **MolMR (Solvent)** | 0.284 | Dispersion force (δd) |
| 3 | Mw (Solvent) | 0.256 | Molecular weight |
| 4 | Mw (Polymer) | 0.220 | Molecular weight |
| 5 | **TPSA (Solvent)** | 0.203 | Polarity (δp) |
| 6 | **TPSA (Polymer)** | 0.160 | Polarity (δp) |
| 7 | BertzCT (Solvent) | 0.108 | Complexity |
| 8 | MolMR (Polymer) | 0.101 | Dispersion force (δd) |

### Functional Group Analysis (cf. Fig 5.4b, 5.5a)

Key findings:
- **Ketone** in Polymer: r(χ) = +0.24 — increases χ, captured by TPSA/HBA
- **Benzene** in Polymer: r(χ) = +0.14 — increases χ, captured by MolMR/BertzCT  
- **Benzene** in Solvent: r(χ) = -0.22 — decreases χ, captured by MolMR
- **Halogen** in Solvent: r(χ) = -0.24 — decreases χ (good solvent), captured by MolMR
- **Alcohol** in Solvent: r(χ) = +0.33 — increases χ, captured by TPSA/HBD

Two-level interpretation bridge:
> **Chemical level** (functional groups) → **Descriptor level** (SHAP) → **Prediction** (χ)

## Active Learning

```bash
python -m experiments.train_active
```

Compares three sampling strategies:
- **Random** sampling (baseline)
- **MC Dropout** — uncertainty sampling via dropout variance
- **Ensemble** — uncertainty sampling via model disagreement

The advantage of uncertainty sampling over random sampling is marginal on this dataset due to homogeneity of polymer-solvent data vs. block copolymer data used in the thesis.

## Training

```bash
python -m experiments.train_baseline              # Baseline
python -m experiments.train_physics                # Physics regularization
```

Hyperparameters (matching thesis Section 5.2.3.1):
- Optimizer: AdamW (lr=1e-3, weight_decay=0.01)
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Gradient clipping: 1.0
- Batch size: 8
- Dropout: 0.1
- Loss: MSE

## Uncertainty Quantification

```bash
python -m models.uncertainty              # MC Dropout
python -m models.uncertainty ensemble     # Ensemble (train + inference)
```

## Data Note

The thesis uses block copolymer χ data (1221 points from CHiMaD/PPPdb) as the target task,
which is not publicly downloadable. This reproduction uses polymer-solvent χ data (1586 points from Nistane et al.)
as a substitute. The active learning methodology is dataset-agnostic.

## References

- Xu, H. (2025). 小数据集场景下的机器学习方法辅助聚合物材料设计与研究. 浙江大学博士论文.
- Nistane, D. et al. (2022). *Estimation of the Flory-Huggins interaction parameter...* MRS Advances.
- Maruyama, M. et al. (2023). *Multitask Machine Learning to Predict Polymer–Solvent Miscibility...* Macromolecules.
