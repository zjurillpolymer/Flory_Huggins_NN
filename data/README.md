# Datasets for Flory-Huggins χ Parameter Prediction

## 1. Nistane2022_polymer_solvent_chi.csv

| Item | Value |
|------|-------|
| **Source** | Nistane et al. (2022), *Estimation of the Flory-Huggins interaction parameter of polymer-solvent mixtures using machine learning* |
| **Data points** | 1,586 |
| **Unique polymers** | 58 |
| **Unique solvents** | 140 |
| **Temperature range** | 273 – 526 K |
| **χ range** | −0.55 – 4.11 |
| **Target variable** | `chi` (Flory-Huggins interaction parameter) |
| **Key columns** | `Polymer`, `Polymer SMILES`, `Solvent`, `Solvent SMILES`, `chi`, `temperature`, `Reference Source` |
| **Use in thesis** | Transfer learning pre-training (Section 5.2.2.1) |
| **Measurement methods** | Osmometry, sorption experiments, inverse gas chromatography |

---

## 2. data_Chi.csv

| Item | Value |
|------|-------|
| **Source** | MTL_ChiParameter (Yoshida-lab), *Predicting polymer-solvent miscibility using machine-learned Flory-Huggins interaction parameters* (Macromolecules 2023) |
| **Data points** | 1,190 |
| **χ range** | −2.24 – 4.40 |
| **Polymer classes** | cellulose, polyacrylate, polychlorox, polyester, polyether, polyethylene, polyisobutylene, polypropylene, polystyrenes, polyvinylx, polyxacrylamide |
| **Key columns** | `ps_pair` (polymer-SMILES_solvent-SMILES), `temp`, `chi`, `polymer_class` |
| **Note** | SMILES strings use explicit hydrogen notation (e.g., `[H]/C1=C(/Cl)...`) |
| **Use** | Alternative polymer-solvent χ dataset for model training/active learning |

---

## 3. data_COSMO.csv

| Item | Value |
|------|-------|
| **Source** | COSMO-RS simulations via MTL_ChiParameter (Yoshida-lab) |
| **Data points** | 1,206 |
| **χ range** | −1.97 – 4.62 |
| **Polymer classes** | cellulose, polyacrylate, polychlorox, polyester, polyether, polyethylene, polyethylenex, polyisobutylene, polypropylene, polystyrenes, polyvinylx, polyxacrylamide |
| **Method** | COSMO-RS (Conductor-like Screening Model for Real Solvents) |
| **Use** | Computational χ values as additional training data or benchmark comparison |

---

## 4. data_PI.csv

| Item | Value |
|------|-------|
| **Source** | MTL_ChiParameter (Yoshida-lab) |
| **Data points** | 1,190 |
| **Key columns** | `ps_pair`, `polymer_class`, `soluble` |
| **Use** | Polymer-solvent miscibility classification (soluble/insoluble) |

---

## 5. desc_Chi.csv / desc_COSMO.csv

| Item | Value |
|------|-------|
| **Size** | 795 descriptor columns per file |
| **Descriptor types** | Element counts (H, C, N, O, F, ...), partial charges, sigma profiles (epsilon, sigma), bond counts, functional group frequencies (fr_*) |
| **Format** | Column naming: `{Polymer|Solvent}_{descriptor_name}` |
| **Use** | Pre-computed COSMO-RS descriptors for polymer and solvent molecules |

---

## Summary for Thesis Reproduction

| Thesis Requirement | Available Dataset |
|---|---|
| Solvent-polymer χ (1,586 points, pre-training) | **Nistane2022_polymer_solvent_chi.csv** ✅ |
| Block copolymer χ (1,221 points, target task) | ❌ (CHiMaD/PPPdb, not publicly downloadable) |
| PI1M monomer library (~1M structures) | ❌ (Git LFS, too large) |

**Alternative reproduction strategies:**
- Use `data_Chi.csv` + `data_COSMO.csv` as the target dataset for active learning
- Compute RDKit molecular descriptors (12 features per molecule as in Section 5.2.3.1) from SMILES strings
- The active learning framework + attention model is methodology-agnostic and works on any χ dataset

---

## References

- Nistane, D. et al. (2022). *Estimation of the Flory-Huggins interaction parameter of polymer-solvent mixtures using machine learning.* MRS Advances, 7, 1031-1036.
- Maruyama, M. et al. (2023). *Multitask Machine Learning to Predict Polymer–Solvent Miscibility Using Flory–Huggins Interaction Parameters.* Macromolecules, 56(17), 6877-6888.
- Ma, R. & Luo, T. (2020). *PI1M: A Benchmark Database for Polymer Informatics.* J. Chem. Inf. Model., 60(10), 4684-4690.
- Xu, H. (2025). *小数据集场景下的机器学习方法辅助聚合物材料设计与研究.* 浙江大学博士学位论文.
