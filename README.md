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
│   ├── rdkit_descriptors.pkl                 # 计算结果 (27维: 24描述符 + 3物理约束)
│   └── schnet_loader.py                      # 3D结构生成 + RDKit特征 + PyG DataLoader
│
├── models/
│   ├── attention_net.py                      # 注意力网络 (Encoder + MultiHeadAttention + Decoder)
│   ├── uncertainty.py                        # 不确定性量化 (MC Dropout + Ensemble)
│   ├── chi_schnet.py                         # SchNet编码器 + MLP → χ (QM-only/Combined/Hansen)
│   ├── best_model.pt                         # 基线模型权重 (MAE=0.244)
│   ├── best_physics.pt                       # 物理约束模型权重 (MAE=0.237)
│   ├── best_schnet.pt                        # SchNet预训练权重 (QM9)
│   └── best_qm.pt                            # QM属性→χ 模型权重 (MAE=0.257)
│
├── experiments/
│   ├── train_baseline.py                     # 基线模型训练
│   ├── train_active.py                       # 主动学习主循环 (MC Dropout / Ensemble vs Random)
│   ├── train_physics.py                      # 物理正则化 Loss 训练
│   ├── train_chi_schnet.py                   # ChiSchNet训练 (QM属性 + RDKit)
│   ├── train_with_temp.py                    # 温度特征 + 单调约束
│   └── eval_transfer.py                      # 跨数据集迁移评估
│
├── figures/
│   ├── summary_*.png/svg                     # 不确定性量化结果
│   ├── active_learning_*.png/svg             # 主动学习对比曲线
│   ├── interpret_*.png/svg                   # SHAP 特征重要性 + 相关热图
│   ├── shap_dependence_*.png/svg             # SHAP 依赖图 (Top 6特征)
│   ├── functional_groups_*.png/svg           # 官能团频率 + FG-χ 相关性
│   └── qm_shap_*.png/svg                     # QM属性 SHAP分析
│
├── ROADMAP.md
└── README.md
```

## Model Performance Comparison

| Model | MAE | R² | Approach |
|-------|-----|----|----------|
| RDKit Baseline (24-dim) | 0.244 | 0.643 | 12 RDKit descriptors × 2 molecules → Attention |
| + Physics regularization | 0.237 | 0.639 | Hansen Loss term (λ=0.1) |
| + Temperature feature | 0.238 | 0.651 | 24-dim + normalized temp |
| MC Dropout (T=50) | **0.231** | **0.759** | Single model, 50 forward passes |
| Ensemble (4 models) | 0.236 | 0.746 | 4 independently trained models |
| **ChiSchNet (QM-only)** | **0.222** | **0.645** | **SchNet QM properties → MLP → χ** |
| ChiSchNet (RDKit+QM) | 0.391 | 0.003 | Combined features (needs tuning) |
| Hansen squared-diff | 0.422 | 0.049 | QM properties → (Δ)² → χ |

### Best Model: ChiSchNet (QM-only)

SchNet (pre-trained on QM9, frozen) → 4 QM properties per molecule → 9-dim MLP → χ

```
Polymer SMILES → RDKit 3D → SchNet → [μ, α, HOMO, LUMO]ₚ
                                              concat(9) → MLP → χ
Solvent SMILES → RDKit 3D → SchNet → [μ, α, HOMO, LUMO]ₛ
                                              Temperature
```

## SchNet QM Properties SHAP Analysis

### QM Feature Importance

| Rank | Feature | \|SHAP\| | Physical meaning |
|------|---------|---------|------------------|
| 1 | **α (Solvent)** | 0.222 | Polarizability → Hansen δd (dispersion) |
| 2 | LUMO (Solvent) | 0.179 | Frontier orbital |
| 3 | HOMO (Solvent) | 0.172 | Frontier orbital |
| 4 | μ (Solvent) | 0.132 | Dipole moment → Hansen δp (polarity) |
| 5 | α (Polymer) | 0.128 | Polarizability → Hansen δd (dispersion) |
| 6 | HOMO (Polymer) | 0.064 | Frontier orbital |
| 7 | μ (Polymer) | 0.055 | Dipole moment |
| 8 | LUMO (Polymer) | 0.051 | Frontier orbital |
| 9 | Temperature | 0.036 | |

Key findings:
- **Polarizability (α)** is the most important QM property, validating the physical intuition that dispersion forces dominate χ
- **Solvent properties > Polymer properties** — dataset has 140 solvents vs 58 polymers, so solvent diversity drives more variance
- **Temperature has minimal impact** — consistent with weak temp-χ correlation in this dataset

## Interpretability (Section 5.3.5)

### RDKit SHAP Feature Importance (Top 8)

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

| Functional Group | Polymer r(χ) | Solvent r(χ) | SHAP Descriptor |
|-----------------|-------------|-------------|-----------------|
| Ketone (C=O) | +0.24 | -0.01 | TPSA, HBA |
| Benzene | +0.14 | -0.22 | MolMR, BertzCT |
| Alcohol (-OH) | -0.11 | +0.33 | TPSA, HBD |
| Halogen | +0.01 | -0.24 | MolMR |

Two-level interpretation bridge:
> **Chemical level** (functional groups) → **Descriptor level** (SHAP) → **Prediction** (χ)

## Active Learning

```bash
python -m experiments.train_active
```

Compares three sampling strategies: Random, MC Dropout, Ensemble.
The advantage of uncertainty sampling over random is marginal on this dataset due to its homogeneity.

## Training

```bash
python -m experiments.train_baseline                # RDKit baseline
python -m experiments.train_chi_schnet               # ChiSchNet (best)
python -m experiments.train_physics                  # Physics regularization
python -m experiments.train_with_temp                # Temperature + monotonicity
```

## Uncertainty Quantification

```bash
python -m models.uncertainty              # MC Dropout
python -m models.uncertainty ensemble     # Ensemble (train + inference)
```

## Transfer Learning (Cross-dataset Evaluation)

```bash
python -m experiments.eval_transfer
```

| Target Dataset | MAE | R² |
|---------------|-----|----|
| Nistane (source) | 0.244 | 0.643 |
| MTL-Chi (experimental) | 0.510 | -0.168 |
| COSMO-RS (computed) | 0.495 | -0.399 |

Limited transferability due to differences in SMILES notation and chemical space.

## Data Note

The thesis uses block copolymer χ data (1221 points from CHiMaD/PPPdb) as the target task,
which is not publicly downloadable. This reproduction uses polymer-solvent χ data (1586 points from Nistane et al.)
as a substitute. The active learning methodology is dataset-agnostic.

## References

- Xu, H. (2025). 小数据集场景下的机器学习方法辅助聚合物材料设计与研究. 浙江大学博士论文.
- Nistane, D. et al. (2022). *Estimation of the Flory-Huggins interaction parameter...* MRS Advances.
- Maruyama, M. et al. (2023). *Multitask Machine Learning to Predict Polymer–Solvent Miscibility...* Macromolecules.
- Schütt, K. et al. (2017). *SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.* NeurIPS.
