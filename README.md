# SchemaFlexTabNet-QC — Schema-Flexible Cross-Cohort Heart Disease Risk Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Paper:** *SchemaFlexTabNet-QC: Schema-Flexible Tabular Networks with Clinical Quality Control for Cross-Cohort Heart Disease Risk Prediction*
> **Manuscript:** [`paper_FINAL_submittable.tex`](paper_FINAL_submittable.tex) · **PDF:** [`SchemaFlexTabNet_QC_Research_Paper.pdf`](SchemaFlexTabNet_QC_Research_Paper.pdf)

---

## Key Research Contributions

1. **Forensic Audit of Baseline Flaws** — Documents that the widely-cited Kaggle Heart Disease dataset contains **723 duplicate rows out of 1,025** (70.5%), causing massive data leakage ($97.1\%$ of test samples seen during training, inflating accuracy to $98.5\%$).

2. **Schema-Flexible Clinical Tokenization** — Tokenizes heterogeneous EHR columns without inner joins:
   $$T_{i,j} = \mathbf{e}_{\text{name}}(f_j) + \phi_{\text{val}}(x_{i,j})$$
   Uses masked self-attention and random feature masking ($p_{\text{mask}} \in [0.15, 0.35]$) for multi-cohort training across hospitals with different column subsets.

3. **Clinical Quality Control & Conformal Abstention** — Hemodynamic plausibility checks, post-hoc Temperature Scaling, and Split Conformal Prediction triggering an **"Abstain / Refer to Clinician"** flag on ambiguous cases at guaranteed $1-\alpha$ coverage.

4. **Knowledge-Guided Monotonicity Regularization** — Differentiable gradient penalty enforcing clinically monotonic risk constraints (Age, BP, Cholesterol), reducing violations from $19.23\%$ → $0.26\%$.

---

## Repository Structure

```text
SchemaFlexTabNet-QC/
├── configs/
│   └── schema_registry.yaml              # Canonical feature registry & plausibility bounds
├── data/
│   ├── raw/                              # Immutable source datasets
│   │   ├── heart_kaggle_johnsmith88.csv  # Kaggle duplicated dataset (for audit)
│   │   ├── cardio_sulianova.csv          # Sulianova CVD dataset (70,000 rows)
│   │   ├── framingham.csv               # Framingham Heart Study (4,240 rows)
│   │   ├── heart_uci_*.data             # UCI 4-institution cohorts (920 patients)
│   │   ├── statlog_heart.csv            # UCI Statlog cohort (270 patients)
│   │   └── z_alizadeh_sani.xlsx         # Z-Alizadeh Sani cohort (303 patients)
│   └── processed/
│       ├── clean_multi_cohort.parquet    # Harmonized, plausibility-filtered benchmark
│       ├── clean_multi_cohort.csv        # Same, CSV format
│       └── harmonization_audit_log.csv  # Audit log of removed corrupt/duplicate rows
├── src/
│   ├── data_harmonizer.py               # Plausibility gatekeeper, unit conversion, dedup
│   ├── schema_tokenizer.py              # Multi-cohort schema-flexible tokenizer
│   ├── model.py                         # SchemaFlexTabNet-QC (concept head + monotonicity)
│   ├── conformal_qc.py                  # Temperature scaling & split conformal abstention
│   └── evaluate_metrics.py             # ECE, Brier, DeLong, Bootstrap CI, DCA, Fairness
├── results/                             # All generated figures and benchmark tables
│   ├── figure1_forensic_audit.png
│   ├── figure2_risk_coverage.png
│   ├── figure3_decision_curve.png
│   ├── figure4_fairness_disparity.png
│   ├── figure5_missingness_degradation.png
│   ├── figure6_mondrian_fairness_parity.png
│   ├── figure7_federated_vs_centralized.png
│   ├── table1_forensic_audit.csv … table9_federated_learning.csv
│   └── loco_multi_seed_stability_audit.csv
├── train_loco_pipeline.py               # LOCO training & evaluation pipeline
├── run_ablation_and_loco_matrix.py      # Ablation study & multi-cohort LOCO matrix
├── run_federated_and_fairness_suite.py  # Federated learning & fairness evaluation
├── evaluate_multiseed_loco_and_conformal.py  # Multi-seed LOCO & conformal evaluation
├── benchmark_baselines.py               # 7-model baseline benchmarking (5-fold CV + LOCO)
├── generate_figures.py                  # Publication-grade figures (300 DPI)
├── download_datasets.py                 # Dataset download helper
├── run_all.py                           # Master end-to-end runner
├── paper_FINAL_submittable.tex          # ⭐ Final submission manuscript
└── SchemaFlexTabNet_QC_Research_Paper.pdf  # ⭐ Final PDF
```

---

## Quickstart

### Install Dependencies
```bash
pip install torch lightgbm xgboost catboost scikit-learn pandas numpy matplotlib seaborn
```

### Run Full Pipeline
```bash
python run_all.py
```

### Run Individual Phases
```bash
# Phase 1: Multi-Cohort Harmonization & QC
python src/data_harmonizer.py

# Phase 2: Classical Baseline Benchmarking (7 models, 5-Fold CV + LOCO)
python benchmark_baselines.py

# Phase 3: Proposed Architecture (LOCO, Conformal Abstention)
python train_loco_pipeline.py

# Phase 4: Ablation Study & Multi-Cohort LOCO Matrix
python run_ablation_and_loco_matrix.py

# Phase 5: Multi-Seed LOCO Stability & Conformal Evaluation
python evaluate_multiseed_loco_and_conformal.py

# Phase 6: Federated Learning & Fairness Suite
python run_federated_and_fairness_suite.py

# Generate Publication Figures (300 DPI)
python generate_figures.py
```

---

## Key Results

### Data Harmonization Summary

| Cohort | Initial Rows | Removed | Clean Rows |
| :--- | :---: | :---: | :---: |
| UCI Cleveland | 303 | 0 | **303** |
| UCI Hungarian | 294 | 1 | **293** |
| UCI Switzerland | 123 | 11 | **112** |
| UCI VA Long Beach | 200 | 4 | **196** |
| Sulianova CVD (Kaggle) | 70,000 | 1,432 | **68,568** |
| Framingham Heart Study | 4,240 | 2 | **4,238** |
| UCI Statlog Heart | 270 | 0 | **270** |
| Z-Alizadeh Sani | 303 | 0 | **303** |
| *Kaggle Heart (johnsmith88)* | *1,025* | ***723 duplicates*** | ***302*** |

### Table 1 — Forensic Audit of Baseline Flaws

| Setup | AUROC | Accuracy | Diagnosis |
| :--- | :---: | :---: | :--- |
| Baseline (Duplicated + Leaked Split) | 0.998 | 98.54% | ❌ INVALID — 97.1% test rows seen in training |
| Forensic Clean (Deduplicated) | 0.859 | 77.05% | ✅ VALID — realistic clinical performance |
| Naive Inner-Join Merge | 0.794 | 73.21% | ✅ VALID — matches theoretical ceiling |

### Table 3 — Zero-Shot Transfer (UCI Statlog Hold-Out)

| Setup | Coverage | AUROC [95% CI] | Accuracy | Brier | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SchemaFlexTabNet-QC (Full) | 100.0% | 0.8821 [0.839–0.918] | 78.15% | 0.1454 | 0.0768 |
| SchemaFlexTabNet-QC (Conformal α=0.10) | **65.6%** | **0.9241** | **88.14%** | **0.1028** | **0.0412** |

### Table 5 — Component Ablation

| Variant | AUROC | Accuracy | Brier | Monotonicity Violations |
| :--- | :---: | :---: | :---: | :---: |
| 1. Inner-Join Baseline | 0.6342 | 59.73% | 0.2532 | 15.04% |
| 2. + Schema-Flexible Tokenizer | 0.8560 | 75.77% | 0.1872 | 19.23% |
| 3. + Random Feature Dropout | 0.8604 | 77.13% | 0.1540 | 4.18% |
| 4. + Concept Head | 0.8611 | 77.82% | 0.1582 | 12.97% |
| **5. Full SchemaFlexTabNet-QC** | **0.8652** | **80.20%** | **0.1422** | **0.26%** |

### Table 6 — Missingness Stress Test

| Missing Rate | SchemaFlexTabNet-QC | LightGBM | Advantage |
| :---: | :---: | :---: | :---: |
| 0% | **0.8520** | 0.7591 | +0.0929 |
| 20% | **0.8036** | 0.6752 | +0.1284 |
| 50% | **0.7123** | 0.5624 | **+0.1499** |

### Table 7 — DeLong Statistical Significance

| Held-Out Cohort | Proposed AUROC | vs. HybridTabNet p-value | vs. LightGBM p-value |
| :--- | :---: | :---: | :---: |
| UCI Hungarian (N=293) | **0.8641** | 2.22×10⁻¹⁶ (***) | 1.04×10⁻¹⁴ (***) |
| UCI VA + Switzerland (N=308) | **0.7241** | 4.51×10⁻⁴ (***) | 4.07×10⁻⁵ (***) |
| UCI Cleveland Dedup (N=303) | **0.8433** | 5.65×10⁻⁷ (***) | 7.99×10⁻¹⁵ (***) |
| Z-Alizadeh Sani (N=303) | 0.7043 | 0.661 (n.s.) | 0.339 (n.s.) |
| Framingham (N=4,238) | 0.6778 | 0.315 (n.s.) | 0.134 (n.s.) |

### Table 9 — Simulated Cross-Hospital Federated Learning

| Framework | Privacy | Transfer AUROC | Gap to Centralized |
| :--- | :--- | :---: | :---: |
| Isolated Local Models | Local only | 0.7461 | −0.1286 |
| **Schema-Aware FedAvg (10 rounds)** | **Zero row sharing** | **0.7671** | **−0.1075** |
| Centralized Pooled Training | Full data sharing | 0.8746 | 0.0000 |

---

## Citation

If you use this code or results, please cite:

```bibtex
@article{schemaflextabnet2026,
  title   = {SchemaFlexTabNet-QC: Schema-Flexible Tabular Networks with Clinical Quality
             Control for Cross-Cohort Heart Disease Risk Prediction},
  author  = {Paida, Rathan and {et al.}},
  year    = {2026},
  journal = {Under Review}
}
```

---

## License

This project is licensed under the MIT License.
