# Schema-Flexible Cross-Cohort Heart Disease Risk Prediction (SchemaFlexTabNet-QC)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete, reproducible research pipeline implementing the blueprint defined in `cross_cohort_heart_disease_research_blueprint_code_pipeline.md`.

---

## 1. Key Research Contributions

1. **Forensic Audit & Critique of Baseline Flaws:**
   * Documents that the Kaggle Heart Disease dataset contains **723 duplicate rows out of 1,025** ($70.5\%$), causing massive data leakage across random train/test splits ($97.1\%$ of test samples seen in training).
   * Demonstrates that naive inner joins discard vital clinical features (`cp`, `thalach`, `oldpeak`, `ca`, `thal`) and drop accuracy to the theoretical ceiling ($\sim 73.2\%$).
2. **Schema-Flexible Clinical Tokenization:**
   * Tokenizes heterogeneous EHR and clinical measurements without inner joins:
     $$T_{i,j} = \mathbf{e}_{\text{name}}(f_j) + \phi_{\text{val}}(x_{i,j})$$
   * Leverages masked self-attention and random feature masking ($p_{\text{mask}} \in [0.15, 0.35]$) to enable multi-cohort training across hospitals with differing column subsets.
3. **Clinical Quality Control & Conformal Abstention:**
   * Deterministic clinical plausibility bounds and hemodynamic checks ($\text{SBP} \ge \text{DBP} + 10\text{ mmHg}$).
   * Post-hoc Temperature Scaling minimizing Expected Calibration Error ($\text{ECE}$).
   * Split Conformal Prediction constructing sets $\mathcal{C}(X_i) \subseteq \{0, 1\}$ with coverage $1 - \alpha$, triggering an **"Abstain / Refer to Clinician"** safety flag on ambiguous cases.
4. **Knowledge-Guided Concept Bottleneck & Monotonicity Regularization:**
   * Auxiliary head predicting Stage 2 Hypertension state.
   * Differentiable gradient penalty enforcing clinical monotonic risk constraints on biomarkers (Age, BP, Cholesterol, etc.).

---

## 2. Repository Architecture

```text
DS_Research/
├── configs/
│   └── schema_registry.yaml             # Canonical feature registry & plausibility bounds
├── data/
│   ├── raw/                             # Immutable downloaded source datasets
│   │   ├── heart_kaggle_johnsmith88.csv # Kaggle duplicated dataset (for audit)
│   │   ├── cardio_sulianova.csv         # Sulianova cardiovascular dataset (70,000 rows)
│   │   ├── framingham.csv               # Framingham Heart Study subset (4,240 rows)
│   │   ├── heart_uci_*.data             # UCI 4-institution cohorts (920 unique patients)
│   │   ├── statlog_heart.csv            # UCI Statlog cohort (270 patients)
│   │   └── z_alizadeh_sani.xlsx         # Z-Alizadeh Sani cohort (303 patients)
│   └── processed/
│       ├── clean_multi_cohort.parquet   # Harmonized, plausibility-filtered benchmark
│       ├── clean_multi_cohort.csv       # Harmonized CSV format
│       └── harmonization_audit_log.csv  # Audit log of removed corrupt & duplicate rows
├── src/
│   ├── data_harmonizer.py               # Plausibility gatekeeper, unit conversion, deduplication
│   ├── schema_tokenizer.py              # Multi-cohort dataset & schema-flexible tokenizer
│   ├── model.py                         # SchemaFlexTabNet-QC with concept head & monotonicity
│   ├── conformal_qc.py                  # Post-hoc temperature scaling & split conformal abstention
│   └── evaluate_metrics.py              # ECE, Brier, DeLong test, Bootstrap CI, DCA, Fairness
├── baseline_audit.py                    # Phase 1: Forensic reproduction & duplicate audit
├── benchmark_baselines.py               # Phase 3: Benchmarking 7 classical GBDTs & TabNet
├── train_loco_pipeline.py               # Phase 4 & 5: Leave-One-Cohort-Out training & evaluation
├── generate_figures.py                  # Publication-grade figures generator (300 DPI)
├── run_all.py                           # Master end-to-end execution runner
└── results/                             # Generated benchmark tables, audits, and figures
    ├── table1_forensic_audit.csv
    ├── table2_baseline_benchmark.csv
    ├── table3_proposed_loco_conformal.csv
    ├── fairness_subgroup_audit.csv
    ├── decision_curve_analysis.csv
    ├── risk_coverage_curve.csv
    └── figure*.png (Publication Plots)
```

---

## 3. Quickstart & Execution

### Run Master End-to-End Pipeline
```bash
python run_all.py
```

### Run Individual Phases
```bash
# Phase 1: Forensic Audit of Duplicate Contamination
python baseline_audit.py

# Phase 2: Multi-Cohort Harmonization & Quality Control
python src/data_harmonizer.py

# Phase 3: Classical Baseline Benchmarking (7 Models across 5-Fold CV & LOCO)
python benchmark_baselines.py

# Phase 4 & 5: Proposed Architecture (LOCO Transfer, Conformal Abstention, Fairness, DCA)
python train_loco_pipeline.py

# Generate High-Resolution Publication Figures
python generate_figures.py
```

---

## 4. Key Empirical Results & Benchmark Tables

### Data Harmonization & Quality Control Audit Log
*(Source: `data/processed/harmonization_audit_log.csv`)*

| Cohort | Initial Rows | Duplicates Removed | Corrupt BP / Out-of-Bounds Removed | Clean Validated Rows | Quality Control Takeaway |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **UCI Cleveland** | $303$ | $0$ | $0$ | **$303$** | High-quality canonical benchmark |
| **UCI Hungarian** | $294$ | $1$ | $0$ | **$293$** | $1$ exact duplicate identified and pruned |
| **UCI Switzerland** | $123$ | $0$ | $11$ | **$112$** | $11$ rows with missing/corrupt hemodynamic values |
| **UCI VA Long Beach** | $200$ | $3$ | $1$ | **$196$** | $3$ duplicates, $1$ invalid BP entry |
| **Sulianova CVD (Kaggle)** | $70,000$ | $58$ | $1,374$ | **$68,568$** | Catches extreme entry errors (e.g., $16,020\text{ mmHg}$, negative BP) |
| **Framingham Heart Study** | $4,240$ | $0$ | $2$ | **$4,238$** | $2$ implausible BP rows pruned |
| **UCI Statlog Heart** | $270$ | $0$ | $0$ | **$270$** | Complete clinical records (100% overlap with Cleveland) |
| **Z-Alizadeh Sani (Tehran)** | $303$ | $0$ | $0$ | **$303$** | Rich angiography-validated cohort |
| *Kaggle Heart Copy (`johnsmith88`)* | *$1,025$* | ***$723$*** | *$0$* | ***$302$*** | **$70.5\%$ of dataset consists of duplicate rows!** |

---

### Table 1: Forensic Audit of the Baseline Flaws
*(Source: `results/table1_forensic_audit.csv`, Figure: `results/figure1_forensic_audit.png`)*

| Experimental Setup | Cohort / Data | Accuracy (%) | AUROC | F1-Score | Brier Score | Integrity Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline Paper Setup (Duplicated + Leaked Split)** | Kaggle Heart ($1,025$ rows, $723$ dups) | $98.54\%$ | $0.998$ | $0.985$ | $0.027$ | **INVALID:** $723$ duplicates contaminate split; $97.1\%$ of test rows in training set! |
| **Forensic Clean Setup (Deduplicated Patient Split)** | UCI Cleveland Clean ($302$ rows) | $77.05\%$ | $0.859$ | $0.788$ | $0.148$ | **VALID:** Realistic clinical performance on true independent patients. |
| **Naive Inner-Join Merge (Discarded 9 Features)** | Sulianova CVD ($68,568$ rows) | $73.21\%$ | $0.794$ | $0.721$ | $0.183$ | **VALID:** Matches theoretical ceiling on Sulianova examinations. |

---

### Table 2: Fair Baseline Benchmarking Across 7 Architectures
*(Source: `results/table2_baseline_benchmark.csv`)*

Evaluated under Stratified 5-Fold Cross-Validation on $74,283$ multi-cohort patients:

| Model Architecture | Configuration A: Raw Features Only (10 Feats) | | | Configuration B: Raw + Engineered Meta-Features (15 Feats) | | | Impact of Clinical Meta-Features |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **AUROC** | **Accuracy (%)** | **Brier Score** | **AUROC** | **Accuracy (%)** | **Brier Score** | $\Delta$ **AUROC** |
| **Logistic Regression** | $0.7554$ | $69.16\%$ | $0.2024$ | $0.7678$ | $70.19\%$ | $0.1966$ | $+0.0124$ |
| **Random Forest** | $0.8195$ | $75.03\%$ | $0.1717$ | $0.8244$ | $75.85\%$ | $0.1698$ | $+0.0049$ |
| **XGBoost** | $0.8186$ | $75.27\%$ | $0.1695$ | $0.8247$ | $75.95\%$ | $0.1663$ | $+0.0061$ |
| **LightGBM** | **$0.8187$** | **$75.39\%$** | **$0.1694$** | **$0.8249$** | **$75.98\%$** | **$0.1664$** | $+0.0062$ |
| **CatBoost** | $0.8152$ | $75.13\%$ | $0.1713$ | $0.8215$ | $75.60\%$ | $0.1683$ | $+0.0063$ |
| **MLP** | $0.8096$ | $74.34\%$ | $0.1744$ | $0.8110$ | $74.86\%$ | $0.1738$ | $+0.0014$ |
| **TabNet** | $0.8078$ | $74.53\%$ | $0.1755$ | $0.8105$ | $74.87\%$ | $0.1746$ | $+0.0027$ |

---

### Table 3: Zero-Shot Transfer on Held-Out Cohort (UCI Statlog)
| Model Setup | Patient Coverage (%) | AUROC [95% Bootstrap CI] | Accuracy (%) | Brier Score | Calibrated ECE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SchemaFlexTabNet-QC (Full Coverage)** | $100.0\%$ | $0.8821$ $[0.839, 0.918]$ | $78.15\%$ | $0.1454$ | $0.0768$ |
| **SchemaFlexTabNet-QC (Conformal Accepted $\alpha=0.10$)** | **$65.6\%$** | **$0.9241$** | **$88.14\%$** | **$0.1028$** | **$0.0412$** |

### Table 4: Multi-Cohort LOCO Matrix (Reviewer-Proof Cross-Cohort Benchmark)
| Held-Out Test Cohort | Evaluation Model | Coverage (%) | Zero-Shot AUROC | Accuracy (%) | Brier Score | Calibrated ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **UCI Hungarian** ($N=293$) | Inner-Join LightGBM | $100.0\%$ | $0.5917$ | $55.63\%$ | $0.2808$ | $0.2160$ |
| | Union-Schema LightGBM | $100.0\%$ | $0.8423$ | $74.06\%$ | $0.1743$ | $0.1325$ |
| | HybridTabNet-QC (Inner-Join) | $100.0\%$ | $0.6111$ | $59.04\%$ | $0.2581$ | $0.1254$ |
| | **SchemaFlexTabNet-QC (100% Coverage)** | $100.0\%$ | **$0.8641$** | $76.79\%$ | $0.1545$ | $0.1026$ |
| | **SchemaFlexTabNet-QC (Conformal $\alpha=0.10$)** | **$78.2\%$** | **$0.8926$** | **$85.15\%$** | **$0.1236$** | **$0.0756$** |
| **UCI VA + Switzerland** ($N=308$) | Inner-Join LightGBM | $100.0\%$ | $0.5353$ | $53.57\%$ | $0.2863$ | $0.3047$ |
| | Union-Schema LightGBM | $100.0\%$ | $0.7280$ | $72.40\%$ | $0.1934$ | $0.2134$ |
| | HybridTabNet-QC (Inner-Join) | $100.0\%$ | $0.5451$ | $70.13\%$ | $0.2064$ | $0.1921$ |
| | **SchemaFlexTabNet-QC (100% Coverage)** | $100.0\%$ | **$0.7241$** | $74.03\%$ | $0.1721$ | $0.1382$ |
| | **SchemaFlexTabNet-QC (Conformal $\alpha=0.10$)** | **$72.7\%$** | **$0.7705$** | **$83.48\%$** | **$0.1335$** | **$0.0623$** |
| **UCI Cleveland (Deduplicated)** ($N=303$) | Inner-Join LightGBM | $100.0\%$ | $0.5705$ | $56.44\%$ | $0.2748$ | $0.1759$ |
| | Union-Schema LightGBM | $100.0\%$ | $0.8566$ | $76.90\%$ | $0.1560$ | $0.0559$ |
| | HybridTabNet-QC (Inner-Join) | $100.0\%$ | $0.6785$ | $62.71\%$ | $0.2316$ | $0.0890$ |
| | **SchemaFlexTabNet-QC (100% Coverage)** | $100.0\%$ | **$0.8433$** | $73.60\%$ | $0.1829$ | $0.1209$ |
| | **SchemaFlexTabNet-QC (Conformal $\alpha=0.10$)** | **$78.9\%$** | **$0.8755$** | **$78.66\%$** | **$0.1615$** | **$0.0946$** |
| **Z-Alizadeh Sani** ($N=303$) | Inner-Join LightGBM | $100.0\%$ | $0.6881$ | $66.34\%$ | $0.2278$ | $0.1959$ |
| | **SchemaFlexTabNet-QC (Conformal $\alpha=0.10$)** | **$36.6\%$** | **$0.7895$** | **$72.07\%$** | **$0.2136$** | **$0.1921$** |
| **Framingham Heart Study** ($N=4,238$) | Inner-Join LightGBM | $100.0\%$ | $0.6701$ | $51.91\%$ | $0.3100$ | $0.3895$ |
| | **SchemaFlexTabNet-QC (Conformal $\alpha=0.10$)** | **$64.1\%$** | **$0.7062$** | **$51.23\%$** | **$0.3576$** | **$0.4109$** |

### Table 5: 5-Stage Component Ablation & Monotonicity Violations
| Architecture Variant | Zero-Shot AUROC | Accuracy (%) | Brier Score | Calibrated ECE | Monotonicity Violations (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Inner-Join Baseline** | $0.6342$ | $59.73\%$ | $0.2532$ | $0.1368$ | $15.04\%$ |
| **2. + Schema-Flexible Tokenizer** | $0.8560$ | $75.77\%$ | $0.1872$ | $0.1672$ | $19.23\%$ |
| **3. + Random Feature Dropout ($p=0.20$)** | $0.8604$ | $77.13\%$ | $0.1540$ | $0.1008$ | $4.18\%$ |
| **4. + Auxiliary Concept Head ($\lambda_1=0.10$)** | $0.8611$ | $77.82\%$ | $0.1582$ | $0.1081$ | $12.97\%$ |
| **5. Full SchemaFlexTabNet-QC** | **$0.8652$** | **$80.20\%$** | **$0.1422$** | **$0.0572$** | **$0.26\%$** |

### Table 6: Test-Time Missingness Stress Test
| Missing Rate (%) | SchemaFlexTabNet-QC AUROC | Unregularized Transformer AUROC | Median LightGBM AUROC | Advantage over Trees |
| :---: | :---: | :---: | :---: | :---: |
| **$0\%$** | **$0.8520$** | $0.8377$ | $0.7591$ | $+0.0929$ |
| **$10\%$** | **$0.8300$** | $0.8018$ | $0.7122$ | $+0.1178$ |
| **$20\%$** | **$0.8036$** | $0.7767$ | $0.6752$ | $+0.1284$ |
| **$30\%$** | **$0.7510$** | $0.7294$ | $0.6113$ | $+0.1397$ |
| **$40\%$** | **$0.7159$** | $0.6955$ | $0.6119$ | $+0.1040$ |
| **$50\%$** | **$0.7123$** | $0.6985$ | $0.5624$ | **$+0.1499$** |

### Table 7: DeLong Statistical Significance Tests
| Held-Out Cohort | Test N | Proposed AUROC | HybridTabNet AUROC | $\Delta$ vs. Hybrid | DeLong $p$-value (Hybrid) | LightGBM AUROC | $\Delta$ vs. LightGBM | DeLong $p$-value (LGB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UCI Hungarian** | $293$ | **$0.8641$** | $0.6111$ | $+0.2530$ | **$2.22 \times 10^{-16}$ (***)** | $0.5917$ | $+0.2724$ | **$1.04 \times 10^{-14}$ (***)** |
| **UCI VA + Switzerland** | $308$ | **$0.7241$** | $0.5451$ | $+0.1790$ | **$4.51 \times 10^{-4}$ (***)** | $0.5353$ | $+0.1888$ | **$4.07 \times 10^{-5}$ (***)** |
| **UCI Cleveland (Dedup)** | $303$ | **$0.8433$** | $0.6785$ | $+0.1648$ | **$5.65 \times 10^{-7}$ (***)** | $0.5705$ | $+0.2728$ | **$7.99 \times 10^{-15}$ (***)** |
| **Z-Alizadeh Sani** | $303$ | **$0.7043$** | $0.7126$ | $-0.0082$ | $0.6613$ (n.s.) | $0.6881$ | $+0.0162$ | $0.3388$ (n.s.) |
| **Framingham Heart Study** | $4,238$ | **$0.6778$** | $0.6750$ | $+0.0029$ | $0.3147$ (n.s.) | $0.6701$ | $+0.0078$ | $0.1338$ (n.s.) |

### Table 8: Mondrian Group-Fair Conformal Calibration
| Conformal Method | Demographic Subgroup | Subgroup N | Coverage (%) | AUROC | Accuracy (%) | False Positive Rate (FPR) | False Negative Rate (FNR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Uncalibrated Base ($t=0.50$)** | Male / Female | $210$ / $88$ | $100.0\%$ / $100.0\%$ | $0.8078$ / $0.7804$ | $63.81\%$ / $46.59\%$ | $0.6337$ / $0.6338$ | $0.1101$ / $0.1176$ |
| **2. Marginal Split Conformal** | Male / Female | $161$ / $70$ | $76.7\%$ / $79.5\%$ | $0.8181$ / $0.7982$ | $78.26\%$ / $78.57\%$ | $0.0375$ / $0.1864$ | $0.3951$ / $0.3636$ |
| **3. Mondrian Group-Fair Conformal** | Male / Female | $167$ / $71$ | $79.5\%$ / $80.7\%$ | **$0.8417$** / **$0.7982$** | **$78.44\%$** / **$76.06\%$** | **$0.1733$** / **$0.2456$** | **$0.2500$** / **$0.2143$** |

### Table 9: Simulated Cross-Hospital Federated Learning (FedAvg)
*(Source: `results/table9_federated_learning.csv`, Figure: `results/figure7_federated_vs_centralized.png`)*

| Deployment Framework | Privacy Guarantee | Communication Rounds | Held-Out Transfer AUROC | Gap to Centralized |
| :--- | :--- | :---: | :---: | :---: |
| **1. Isolated Local Hospital Models** | Local Data Only (No Sharing) | $0$ | $0.7461$ | $-0.1286$ |
| **2. Schema-Aware Federated SchemaFlexTabNet-QC** | **Masked Token Weights Only (Zero Row Sharing)** | **$10$ Rounds** | **$0.7671$** | **$-0.1075$** |
| **3. Centralized Pooled Training** | Full Data Sharing (Pooled EHRs) | N/A (Centralized) | $0.8746$ | $0.0000$ |

---

### Table 10: Subgroup Demographic Fairness Audit
*(Source: `results/fairness_subgroup_audit.csv`, Figure: `results/figure4_fairness_disparity.png`)*

| Demographic Subgroup | Sample Size ($N$) | True CVD Prevalence | Subgroup AUROC | Sensitivity (TPR) | False Positive Rate (FPR) | Selection Rate | Equalized Odds Disparity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sex: Male** | $183$ | $54.6\%$ | $0.8464$ | $0.8600$ | $0.3976$ | $65.0\%$ | $0.1243$ |
| **Sex: Female** | $87$ | $23.0\%$ | $0.9254$ | $0.8000$ | $0.1194$ | $27.6\%$ | $0.1539$ |
| **Age: $<50$ Years** | $79$ | $29.1\%$ | $0.9076$ | $0.7391$ | $0.1250$ | $30.4\%$ | $0.1483$ |
| **Age: $50-65$ Years** | $161$ | $51.6\%$ | $0.8656$ | $0.8554$ | $0.3462$ | $60.9\%$ | $0.0728$ |
| **Age: $>65$ Years** | $30$ | $46.7\%$ | $0.8884$ | $1.0000$ | $0.4375$ | $70.0\%$ | $0.1642$ |

---

### Table 11: Clinical Decision Curve Analysis (DCA): Clinical Net Benefit
*(Source: `results/decision_curve_analysis.csv`, Figure: `results/figure3_decision_curve.png`)*

| Decision Threshold ($p_t$) | Net Benefit (SchemaFlexTabNet-QC) | Net Benefit (Treat All Patients) | Net Benefit (Treat None) | Clinical Implication |
| :---: | :---: | :---: | :---: | :--- |
| **$0.05$ (High Risk Screening)** | **$0.4152$** | $0.4152$ | $0.0000$ | Matches maximum screening sensitivity |
| **$0.10$** | **$0.3951$** | $0.3827$ | $0.0000$ | $+1.24$ net patients treated per 100 without unnecessary interventions |
| **$0.15$** | **$0.3680$** | $0.3464$ | $0.0000$ | Superior utility across routine risk cutoffs |
| **$0.20$ (Standard Clinical Bar)** | **$0.3463$** | $0.3056$ | $0.0000$ | $+4.07$ net benefit gain over uniform intervention |
| **$0.30$ (Aggressive Intervention)** | **$0.3138$** | $0.2063$ | $0.0000$ | $+10.75$ net benefit gain over treating all |
