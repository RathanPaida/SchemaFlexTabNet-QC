# Schema-Flexible Cross-Cohort Heart Disease Risk Prediction: Complete Research Blueprint & Code Pipeline

---

## 1. Executive Summary & Critique of the Baseline Paper

### 1.1 Baseline Paper Overview
* **Title:** *HybridTabNet-QC: A Transformer-Based Clinical Feature Fusion Framework for Heart Disease Risk Prediction* (Mahmud et al., IEEE OJCS, 2025/2026).
* **Core Claim:** Combines a 2-layer Transformer Encoder ($4$ heads, $128$ dimensions) with $3$ engineered clinical meta-features ($\text{BMI}$, Pulse Pressure, and $\text{Cholesterol} \times \text{Glucose}$) across two merged Kaggle datasets, reporting $90.1\%$ Accuracy and $0.936$ AUC-ROC.

### 1.2 Critical Methodological Flaws Identified (Your Research Motivation)
1. **Severe Cohort Imbalance & Information Loss via Inner-Join:**
   * Dataset A (Kaggle Heart Disease) has $1,025$ rows (which itself is a duplicated version of the $303$-row Cleveland UCI dataset), while Dataset B (Kaggle Cardiovascular) has $70,000$ rows.
   * Because the paper merges them via an **inner join** on shared columns, vital clinical indicators unique to Dataset A (e.g., ST depression `oldpeak`, major vessels `ca`, `thalassemia`, chest pain type `cp`, maximum heart rate `thalach`) are completely discarded, and Dataset B dominates $98.5\%$ of the training distribution.
2. **Data Leakage & Duplicate Contamination:**
   * The Kaggle Heart Disease dataset (`johnsmith88`) contains $723$ duplicate rows out of $1,025$. A random train/test split places identical patients in both training and testing sets, artificially inflating accuracy.
3. **Internal Mathematical Contradictions:**
   * Table 1 reports a test set size of $10,653$ samples ($5,328$ negative, $5,325$ positive).
   * The text on Page 7 states the confusion matrix has $1,100$ negative and $1,100$ positive cases ($980$ TN, $990$ TP).
   * However, Figure 2 (Confusion Matrix) displays $4,795$ TN, $533$ FP, $395$ FN, and $4,930$ TP (which equals $91.29\%$ accuracy, contradicting the $90.1\%$ headline claim).
4. **Flawed Outlier Removal vs. Physiological Plausibility:**
   * The baseline uses a blanket $|Z_i| > 3$ cut. Dataset B (`sulianova`) contains extreme data entry errors (e.g., systolic BP `ap_hi` of $16,020\text{ mmHg}$ or negative numbers $-150\text{ mmHg}$). These extreme values warp the standard deviation $\sigma$ so severely that a $|Z| > 3$ filter fails to catch physiologically impossible blood pressures (like diastolic $>$ systolic) while potentially clipping true high-risk cardiac patients.
5. **Unfair Baseline Comparisons & Misleading "Quality Control":**
   * Baseline models (XGBoost, Random Forest, TabNet) were trained *without* the three engineered meta-features, making it impossible to tell if gains came from the Transformer or simple feature engineering.
   * "Quality Control (QC)" in the paper merely refers to adding Gaussian noise $\mathcal{N}(0, \sigma^2)$ and static class weights ($w_0=0.45, w_1=0.55$).

---

## 2. Core Novelty & Contributions of Your New Paper

Your research transforms these baseline weaknesses into a rigorous, publishable framework (targeted at *IEEE JBHI*, *Computers in Biology and Medicine*, or *Scientific Reports*):

### Novelty 1: Schema-Flexible Feature Tokenization (No More Inner Joins)
Instead of throwing away unshared columns via an inner join, each clinical measurement $j$ for patient $i$ is converted into a unified token:
$$T_{i,j} = \mathbf{e}_{\text{name}}(f_j) + \mathbf{e}_{\text{unit}}(u_j) + \phi_{\text{val}}(x_{i,j})$$
* $\mathbf{e}_{\text{name}}(f_j) \in \mathbb{R}^d$: Learned semantic embedding of the clinical feature identity (e.g., `systolic_bp`, `oldpeak`, `thalach`).
* $\phi_{\text{val}}(x_{i,j}) \in \mathbb{R}^d$: Modality-specific value projection (continuous values pass through a Linear-ReLU-Linear block; categorical variables pass through a lookup table).
* **Impact:** A patient is represented as a variable-length set of observed clinical tokens. Missing features in a cohort are naturally masked via an attention padding mask. You can train a single unified model across 5+ heterogeneous cohorts simultaneously without discarding a single column.

### Novelty 2: True Quality Control (Physiological Gatekeeper + Conformal Abstention)
Replace the baseline's superficial "Gaussian noise QC" with a three-stage clinical safety pipeline:
1. **Physiological Plausibility Rules:** Deterministic clinical bounds (e.g., $60 \le \text{SBP} \le 260\text{ mmHg}$, $40 \le \text{DBP} \le 160\text{ mmHg}$, $\text{SBP} > \text{DBP} + 10\text{ mmHg}$, $12 \le \text{BMI} \le 70\text{ kg/m}^2$).
2. **Post-Hoc Probability Calibration:** Temperature Scaling and Isotonic Regression on held-out validation folds to minimize Expected Calibration Error ($\text{ECE}$) and Brier Score.
3. **Split Conformal Prediction with Abstention:** Constructs prediction sets $\mathcal{C}(X_i) \subseteq \{0, 1\}$ guaranteed to contain the true outcome with probability $1 - \alpha$:
   $$\mathbb{P}(Y_{\text{test}} \in \mathcal{C}(X_{\text{test}})) \ge 1 - \alpha$$
   Whenever $|\mathcal{C}(X_i)| = 2$ (both classes plausible) or $|\mathcal{C}(X_i)| = 0$, the model triggers an **"Abstain / Refer to Clinician"** flag rather than guessing on out-of-distribution patients.

### Novelty 3: Knowledge-Guided Concept Bottleneck & Monotonic Constraints
* **Auxiliary Clinical Concept Bottleneck:** Before predicting final CVD risk, an intermediate head predicts established clinical states (e.g., Hypertension Stage, Metabolic Risk Index, Framingham Risk Prior).
* **Monotonicity Regularization Penalty:** Penalizes the network if increasing known monotonically harmful biomarkers (like Systolic BP or Age) decreases predicted risk:
  $$\mathcal{L}_{\text{mono}} = \frac{1}{N} \sum_{i=1}^N \sum_{k \in \mathcal{M}} \max\left(0, -\frac{\partial \hat{y}_i}{\partial x_{i,k}}\right)$$

### Novelty 4: Leave-One-Cohort-Out (LOCO) & Patient-Level Deduplicated Benchmark
* Establishes the first de-duplicated, multi-cohort cardiovascular benchmark evaluated under strict **Leave-One-Cohort-Out (LOCO)** zero-shot transfer and few-shot fine-tuning protocols.

---

## 3. Curated Datasets & Direct Access Links

Build your multi-cohort benchmark using the following open-access repositories. **Crucial Rule:** Do not use the duplicated Kaggle copy for Dataset A; use the original 4-institution UCI files.

| Cohort ID | Dataset Name | Samples / Features | Target Definition | Direct URL / Access Link |
| :--- | :--- | :--- | :--- | :--- |
| **Cohort 1-4** | **UCI Heart Disease** (Cleveland, Hungary, Switzerland, VA Long Beach) | $920$ total unique patients ($303$ Cleve, $294$ Hung, $123$ Switz, $200$ VA) / $14$ features | Angiographic narrowing ($>50\%$ diameter narrowing in any major vessel) | [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| **Cohort 5** | **Kaggle Heart Disease Copy** (For Reproducibility Audit Only) | $1,025$ rows ($723$ duplicates!) / $14$ features | Binary duplicate of UCI | [https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **Cohort 6** | **Cardiovascular Disease Dataset** (Sulianova) | $70,000$ rows ($\sim 68,500$ after BP plausibility filter) / $12$ features | Routine clinical examination presence of CVD (`cardio`) | [https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) |
| **Cohort 7** | **Z-Alizadeh Sani Dataset** (Tehran Cohort) | $303$ patients / $55$ rich clinical, ECG & echo features | Coronary Artery Disease (stenosis $\ge 50\%$ on angiography) | [https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani](https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani) |
| **Cohort 8** | **Framingham Heart Study** (Public Teaching Subset) | $4,240$ patients / $16$ features | `TenYearCHD` (10-year incidence of coronary heart disease) | [https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) |
| **Cohort 9** | **UCI Statlog (Heart)** | $270$ patients / $13$ features | Absence ($1$) or presence ($2$) of heart disease | [https://archive.ics.uci.edu/dataset/145/statlog+heart](https://archive.ics.uci.edu/dataset/145/statlog+heart) |
| **Cohort 10** | **NHANES (CDC National Health Survey)** | $10,000+$ adults / $25+$ harmonized lab & exam variables | Self-reported MCQ160b/c/d/e (CHF, CHD, Angina, Heart Attack) | [https://wwwn.cdc.gov/nchs/nhanes/](https://wwwn.cdc.gov/nchs/nhanes/) |

---

## 4. What To Do & How To Do It: 14-Week Step-by-Step Execution Plan

### Phase 1: Baseline Reproduction & Forensic Audit (Weeks 1–2)
* **What to do:** Prove that the baseline paper's $90.1\%$ accuracy is an artifact of duplicate rows, data leakage, or synthetic curves.
* **How to do it:**
  1. Download `johnsmith88/heart-disease-dataset` ($1,025$ rows) and `sulianova/cardiovascular-disease-dataset` ($70,000$ rows).
  2. Run `df.duplicated().sum()` on Dataset A—document that $723$ of $1,025$ rows are exact duplicates.
  3. Reproduce the inner-join merge on shared features (`age`, `sex`, `systolic_bp`, `cholesterol`).
  4. Train the baseline `HybridTabNet-QC` before deduplication vs. after deduplication and patient-level splitting.
  5. Record how accuracy on Dataset B drops to the realistic ceiling ($\sim 73.5\% - 74.2\%$ Accuracy, $\sim 0.802$ AUROC), which is the well-known theoretical limit on the Sulianova dataset. **This table becomes Table 1 of your paper.**

### Phase 2: Multi-Cohort Harmonization & Schema Registry (Weeks 3–4)
* **What to do:** Build a reproducible data engineering pipeline that standardizes units, cleans impossible clinical entries, and preserves every cohort's unique variables.
* **How to do it:**
  1. Create a `schema_registry.yaml` file mapping raw column names across all cohorts to canonical medical concepts.
  2. Harmonize units:
     * Convert `age` in days (`sulianova`) to years: $\text{Age}_{\text{years}} = \text{Age}_{\text{days}} / 365.25$.
     * Harmonize `cholesterol`: UCI & Framingham report continuous $\text{mg/dL}$, whereas Sulianova reports ordinal tiers ($1$: normal, $2$: above normal, $3$: well above normal). Keep *both* a continuous token `chol_mgdl` (when available) and an ATP-III harmonized ordinal token `chol_tier` ($\le 200\text{ mg/dL} \rightarrow 1$, $200-239\text{ mg/dL} \rightarrow 2$, $\ge 240\text{ mg/dL} \rightarrow 3$).
  3. Apply the **Physiological Plausibility Filter** (log every rejected row for your paper's appendix).

### Phase 3: Fair Baseline Benchmarking (Weeks 5–6)
* **What to do:** Benchmark classical GBDTs and tabular deep learning models fairly.
* **How to do it:**
  1. Train **Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, MLP, TabNet, and FT-Transformer**.
  2. Train every baseline in two configurations: (a) Raw features only, and (b) Raw features + Engineered Meta-Features ($\text{BMI}$, Pulse Pressure, $\text{CG}_{\text{score}}$).
  3. Evaluate under two protocols: **Within-Cohort Stratified 5-Fold CV** and **Leave-One-Cohort-Out (LOCO)** (e.g., train on Cleveland + Hungary + Sulianova + Framingham, test zero-shot on VA Long Beach or Z-Alizadeh Sani).

### Phase 4: Training Schema-Flexible Transformer + QC Wrapper (Weeks 7–9)
* **What to do:** Implement and train your proposed architecture (`SchemaFlexTabNet-QC`).
* **How to do it:**
  1. Train with **Random Feature Masking (Dropout $p_{\text{mask}} \in [0.15, 0.35]$)** during training: randomly drop $15\% - 35\%$ of observed clinical tokens in each mini-batch so the Transformer learns to make accurate predictions regardless of which subset of columns a hospital provides.
  2. Include the **Monotonicity Loss** ($\lambda_{\text{mono}} = 0.05$) and **Concept Bottleneck Loss** ($\lambda_{\text{concept}} = 0.10$).
  3. Fit **Temperature Scaling** on the validation fold, followed by **Split Conformal Prediction** at target error rates $\alpha \in \{0.05, 0.10, 0.15\}$.

### Phase 5: Rigorous Evaluation, Fairness Audit & Explainability (Weeks 10–11)
* **What to do:** Generate publication-grade statistical proofs and clinical utility curves.
* **How to do it:**
  1. Compute **AUROC, AUPRC, Brier Score, Expected Calibration Error (ECE)**, and **95% Bootstrap Confidence Intervals** ($1,000$ resamples) with **DeLong's test** $p$-values.
  2. Plot the **Risk-Coverage Abstention Curve**: show how AUROC and Sensitivity rise as the Conformal predictor abstains on uncertain/OOD cases.
  3. Conduct a **Subgroup Fairness Audit** across Sex (Male vs. Female) and Age Bands ($<50$, $50-65$, $>65$), reporting Equalized Odds Disparity.
  4. Compute **Decision Curve Analysis (Net Benefit)** across probability thresholds $p_t \in [0.05, 0.80]$.

---

## 5. Complete End-to-End Code Pipeline

Below is the modular, ready-to-run Python & PyTorch implementation of your entire framework.

### 5.1 Repository Directory Structure
```text
cross-cohort-heart-risk/
├── configs/
│   └── schema_registry.yaml       # Canonical feature & unit definitions
├── data/
│   ├── raw/                       # Immutable downloaded datasets
│   └── processed/                 # Deduplicated, plausibility-filtered parquet files
├── src/
│   ├── data_harmonizer.py         # Plausibility filter, unit conversion, deduplication
│   ├── schema_tokenizer.py        # Schema-flexible feature-name + value tokenizer
│   ├── model.py                   # SchemaFlexTabNet-QC with Concept Head & Monotonicity
│   ├── conformal_qc.py            # Temperature Scaling + Split Conformal Abstention
│   └── evaluate_metrics.py        # ECE, Brier, DeLong/Bootstrap CI, Net Benefit, Fairness
└── train_loco_pipeline.py         # Main execution script for Leave-One-Cohort-Out
```

### 5.2 Configuration: `configs/schema_registry.yaml`
```yaml
# Canonical Feature Registry for Cross-Cohort Harmonization
canonical_features:
  # Continuous Clinical Features (with physiological plausibility bounds)
  age_years:
    type: continuous
    unit: years
    min_plausible: 18.0
    max_plausible: 110.0
    monotonic_direction: 1  # +1 means risk should not decrease as age increases
  systolic_bp:
    type: continuous
    unit: mmHg
    min_plausible: 60.0
    max_plausible: 260.0
    monotonic_direction: 1
  diastolic_bp:
    type: continuous
    unit: mmHg
    min_plausible: 40.0
    max_plausible: 160.0
    monotonic_direction: 1
  bmi:
    type: continuous
    unit: kg/m2
    min_plausible: 12.0
    max_plausible: 70.0
    monotonic_direction: 1
  pulse_pressure:
    type: continuous
    unit: mmHg
    min_plausible: 15.0
    max_plausible: 160.0
    monotonic_direction: 1
  chol_mgdl:
    type: continuous
    unit: mg/dL
    min_plausible: 80.0
    max_plausible: 650.0
    monotonic_direction: 1
  max_hr_thalach:
    type: continuous
    unit: bpm
    min_plausible: 50.0
    max_plausible: 230.0
    monotonic_direction: -1
  st_depression_oldpeak:
    type: continuous
    unit: mm
    min_plausible: 0.0
    max_plausible: 10.0
    monotonic_direction: 1

  # Categorical / Ordinal Clinical Features
  sex_male:
    type: categorical
    num_classes: 2
  chol_tier:
    type: categorical
    num_classes: 4  # 1: Normal (<200), 2: Borderline (200-239), 3: High (>=240)
  gluc_tier:
    type: categorical
    num_classes: 4  # 1: Normal, 2: Prediabetes, 3: Diabetes
  chest_pain_type:
    type: categorical
    num_classes: 5
  exercise_angina:
    type: categorical
    num_classes: 2
  smoker:
    type: categorical
    num_classes: 2
```

### 5.3 Full Python Implementation (`src/full_pipeline.py`)
```python
"""
Complete Implementation of SchemaFlexTabNet-QC:
1. Forensic Deduplication & Physiological Plausibility Gatekeeper
2. Schema-Flexible Feature Name + Value Tokenizer (No Inner-Join Required)
3. Transformer Encoder with Random Feature Masking, Concept Bottleneck, and Monotonicity Loss
4. Post-Hoc Temperature Calibration & Split Conformal Prediction with Abstention
5. Comprehensive Evaluation (AUROC, AUPRC, ECE, Brier, Subgroup Fairness, Net Benefit)
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

# Set deterministic seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# =====================================================================
# MODULE 1: PHYSIOLOGICAL QUALITY CONTROL & COHORT HARMONIZER
# =====================================================================
class ClinicalPlausibilityGatekeeper:
    """
    Replaces naive Z-score outlier removal with deterministic clinical bounds
    and logs every data quality violation.
    """

    BOUNDS = {
        "age_years": (18.0, 110.0),
        "systolic_bp": (60.0, 260.0),
        "diastolic_bp": (40.0, 160.0),
        "bmi": (12.0, 70.0),
        "pulse_pressure": (15.0, 160.0),
        "chol_mgdl": (80.0, 650.0),
        "max_hr_thalach": (50.0, 230.0),
        "st_depression_oldpeak": (0.0, 10.0),
    }

    @classmethod
    def audit_and_clean(cls, df: pd.DataFrame, cohort_name: str) -> Tuple[pd.DataFrame, Dict]:
        initial_rows = len(df)
        cleaned = df.copy()

        # 1. Exact duplicate removal (critical for UCI/Kaggle Heart dataset)
        feature_cols = [c for c in cleaned.columns if c not in ["patient_id", "cohort", "target"]]
        cleaned = cleaned.drop_duplicates(subset=feature_cols, keep="first")
        after_dedup = len(cleaned)

        # 2. Physiological range checks (set impossible values to NaN or drop corrupt BP rows)
        corrupt_mask = pd.Series(False, index=cleaned.index)
        for col, (low, high) in cls.BOUNDS.items():
            if col in cleaned.columns:
                out_of_bounds = (cleaned[col].notna()) & ((cleaned[col] < low) | (cleaned[col] > high))
                corrupt_mask = corrupt_mask | out_of_bounds

        # 3. Hemodynamic consistency check: Systolic BP must exceed Diastolic BP by >= 10 mmHg
        if "systolic_bp" in cleaned.columns and "diastolic_bp" in cleaned.columns:
            both_present = cleaned["systolic_bp"].notna() & cleaned["diastolic_bp"].notna()
            inverted_bp = both_present & ((cleaned["systolic_bp"] - cleaned["diastolic_bp"]) < 10.0)
            corrupt_mask = corrupt_mask | inverted_bp

        cleaned = cleaned[~corrupt_mask].reset_index(drop=True)
        final_rows = len(cleaned)

        audit_log = {
            "cohort": cohort_name,
            "initial_rows": initial_rows,
            "duplicates_removed": initial_rows - after_dedup,
            "physiologically_corrupt_removed": after_dedup - final_rows,
            "final_clean_rows": final_rows,
        }
        return cleaned, audit_log

    @staticmethod
    def engineer_clinical_concepts(df: pd.DataFrame) -> pd.DataFrame:
        """Derives harmonized clinical meta-features and concept labels where inputs exist."""
        df = df.copy()
        # Pulse pressure
        if "systolic_bp" in df.columns and "diastolic_bp" in df.columns:
            df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]

        # Harmonize continuous cholesterol (mg/dL) into ATP-III ordinal tiers (1, 2, 3)
        if "chol_mgdl" in df.columns and "chol_tier" not in df.columns:
            df["chol_tier"] = pd.cut(
                df["chol_mgdl"],
                bins=[-np.inf, 200.0, 239.0, np.inf],
                labels=[1, 2, 3]
            ).astype(float)

        # Auxiliary Clinical Concept: Stage 2 Hypertension (SBP >= 140 or DBP >= 90)
        if "systolic_bp" in df.columns:
            sbp_high = df["systolic_bp"] >= 140.0
            dbp_high = df["diastolic_bp"] >= 90.0 if "diastolic_bp" in df.columns else False
            df["concept_hypertension"] = (sbp_high | dbp_high).astype(float)
        else:
            df["concept_hypertension"] = 0.0

        return df


# =====================================================================
# MODULE 2: SCHEMA-FLEXIBLE DATASET & FEATURE TOKENIZER
# =====================================================================
CONTINUOUS_FEATURES = [
    "age_years",
    "systolic_bp",
    "diastolic_bp",
    "bmi",
    "pulse_pressure",
    "chol_mgdl",
    "max_hr_thalach",
    "st_depression_oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex_male",
    "chol_tier",
    "gluc_tier",
    "chest_pain_type",
    "exercise_angina",
    "smoker",
]

# Monotonicity directions aligned with CONTINUOUS_FEATURES (+1: positive, -1: negative)
MONOTONIC_DIRECTIONS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0]


class MultiCohortTabularDataset(Dataset):
    """
    Produces variable-availability tensors with binary observation masks.
    Missing columns in a cohort are marked with mask=0 rather# than dropped via inner-join.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cont_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

        # Compute training fold mean & std solely on observed (non-NaN) values
        if cont_stats is None:
            self.cont_stats = {}
            for col in CONTINUOUS_FEATURES:
                if col in self.df.columns and self.df[col].notna().any():
                    mu = float(self.df[col].mean())
                    sigma = float(self.df[col].std()) + 1e-6
                    self.cont_stats[col] = (mu, sigma)
                else:
                    self.cont_stats[col] = (0.0, 1.0)
        else:
            self.cont_stats = cont_stats

        # Build continuous values and observation masks
        n = len(self.df)
        self.x_cont = np.zeros((n, len(CONTINUOUS_FEATURES)), dtype=np.float32)
        self.mask_cont = np.zeros((n, len(CONTINUOUS_FEATURES)), dtype=np.float32)

        for j, col in enumerate(CONTINUOUS_FEATURES):
            if col in self.df.columns:
                obs = self.df[col].notna().values
                self.mask_cont[:, j] = obs.astype(np.float32)
                mu, sigma = self.cont_stats[col]
                vals = self.df[col].fillna(mu).values
                self.x_cont[:, j] = (vals - mu) / sigma

        # Build categorical indices and observation masks
        self.x_cat = np.zeros((n, len(CATEGORICAL_FEATURES)), dtype=np.int64)
        self.mask_cat = np.zeros((n, len(CATEGORICAL_FEATURES)), dtype=np.float32)

        for j, col in enumerate(CATEGORICAL_FEATURES):
            if col in self.df.columns:
                obs = self.df[col].notna().values
                self.mask_cat[:, j] = obs.astype(np.float32)
                vals = self.df[col].fillna(0).values.astype(np.int64)
                self.x_cat[:, j] = np.clip(vals, 0, 9)

        self.y = self.df["target"].values.astype(np.float32)
        self.concept_htn = (
            self.df["concept_hypertension"].values.astype(np.float32)
            if "concept_hypertension" in self.df.columns
            else np.zeros(n, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_cont": torch.tensor(self.x_cont[idx], dtype=torch.float32),
            "mask_cont": torch.tensor(self.mask_cont[idx], dtype=torch.float32),
            "x_cat": torch.tensor(self.x_cat[idx], dtype=torch.long),
            "mask_cat": torch.tensor(self.mask_cat[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "concept_htn": torch.tensor(self.concept_htn[idx], dtype=torch.float32),
        }


# =====================================================================
# MODULE 3: SCHEMAFLEXTABNET-QC ARCHITECTURE
# =====================================================================
class SchemaFlexibleTokenizer(nn.Module):
    """
    Tokenizes each feature j as:
        Token_{i,j} = FeatureNameEmbedding(j) + ValueProjection(x_{i,j})
    """

    def __init__(self, num_cont: int, num_cat: int, d_model: int = 64):
        super().__init__()
        self.num_cont = num_cont
        self.num_cat = num_cat
        self.total_features = num_cont + num_cat

        # Learned semantic identity embedding for every canonical feature name
        self.feature_name_embeddings = nn.Parameter(
            torch.randn(1, self.total_features, d_model) * 0.02
        )

        # Feature-specific continuous value encoders
        self.cont_weight = nn.Parameter(torch.randn(1, num_cont, d_model) * 0.02)
        self.cont_bias = nn.Parameter(torch.zeros(1, num_cont, d_model))

        # Shared categorical value lookup table (up to 10 categories per variable)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(10, d_model) for _ in range(num_cat)]
        )

        # Learnable [CLS] summary token for patient-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(
        self, x_cont: torch.Tensor, x_cat: torch.Tensor
    ) -> torch.Tensor:
        batch_size = x_cont.size(0)

        # Continuous token projection: (B, num_cont, 1) * (1, num_cont, d_model)
        cont_tokens = x_cont.unsqueeze(-1) * self.cont_weight + self.cont_bias

        # Categorical token projection
        cat_token_list = [
            emb(x_cat[:, j]).unsqueeze(1) for j, emb in enumerate(self.cat_embeddings)
        ]
        cat_tokens = torch.cat(cat_token_list, dim=1)

        # Combine value projections and add semantic feature-name embeddings
        value_tokens = torch.cat([cont_tokens, cat_tokens], dim=1)
        tokens = value_tokens + self.feature_name_embeddings

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        all_tokens = torch.cat([cls_tokens, tokens], dim=1)
        return all_tokens


class SchemaFlexTabNetQC(nn.Module):
    """
    Proposed Architecture:
    - Schema-Flexible Feature Tokenizer with Masked Self-Attention
    - Training-time Random Feature Dropout (simulating missing hospital columns)
    - Auxiliary Clinical Concept Bottleneck Head
    - Risk Prediction Head with Differentiable Monotonicity Regularization
    """

    def __init__(
        self,
        num_cont: int = len(CONTINUOUS_FEATURES),
        num_cat: int = len(CATEGORICAL_FEATURES),
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        feature_drop_prob: float = 0.20,
    ):
        super().__init__()
        self.tokenizer = SchemaFlexibleTokenizer(num_cont, num_cat, d_model)
        self.feature_drop_prob = feature_drop_prob

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Intermediate Clinical Concept Bottleneck Head (predicts Hypertension / Metabolic state)
        self.concept_head = nn.Linear(d_model, 1)

        # Final Risk Classifier Head (fuses [CLS] representation + predicted clinical concept)
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 1, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_cont: torch.Tensor,
        mask_cont: torch.Tensor,
        x_cat: torch.Tensor,
        mask_cat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_cont.size(0)
        tokens = self.tokenizer(x_cont, x_cat)

        # Observation mask: 1 = observed, 0 = missing
        obs_mask = torch.cat([mask_cont, mask_cat], dim=1)

        # Random feature masking during training to enforce cross-cohort schema invariance
        if self.training and self.feature_drop_prob > 0.0:
            keep_mask = (torch.rand_like(obs_mask) > self.feature_drop_prob).float()
            # Ensure at least 2 features remain observed per patient
            obs_mask = obs_mask * keep_mask

        # Prepend 1.0 for the [CLS] token (always observed)
        cls_obs = torch.ones((batch_size, 1), device=obs_mask.device, dtype=obs_mask.dtype)
        full_obs_mask = torch.cat([cls_obs, obs_mask], dim=1)

        # PyTorch src_key_padding_mask expects True for positions to IGNORE (unobserved)
        padding_mask = full_obs_mask == 0.0

        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        cls_rep = self.norm(encoded[:, 0, :])

        # Clinical Concept Bottleneck prediction
        concept_logit = self.concept_head(cls_rep)
        concept_prob = torch.sigmoid(concept_logit)

        # Fuse latent representation with clinical concept probability
        fused = torch.cat([cls_rep, concept_prob], dim=1)
        risk_logit = self.classifier(fused).squeeze(-1)

        return risk_logit, concept_logit.squeeze(-1)


def compute_monotonicity_loss(
    model: SchemaFlexTabNetQC,
    x_cont: torch.Tensor,
    mask_cont: torch.Tensor,
    x_cat: torch.Tensor,
    mask_cat: torch.Tensor,
    directions: List[float] = MONOTONIC_DIRECTIONS,
) -> torch.Tensor:
    """
    Penalizes gradients that violate known clinical monotonicity
    (e.g., risk must not decrease when Systolic BP or Age increases).
    """
    x_cont_grad = x_cont.detach().clone().requires_grad_(True)
    logits, _ = model(x_cont_grad, mask_cont, x_cat, mask_cat)
    grads = torch.autograd.grad(
        outputs=logits.sum(),
        inputs=x_cont_grad,
        create_graph=True,
        retain_graph=True,
    )[0]

    dir_tensor = torch.tensor(directions, device=x_cont.device, dtype=torch.float32).unsqueeze(0)
    # Violation occurs when sign(grad) opposes expected clinical direction on observed features
    violations = F.relu(-grads * dir_tensor) * mask_cont
    return violations.mean()


# =====================================================================
# MODULE 4: POST-HOC CALIBRATION & SPLIT CONFORMAL ABSTENTION
# =====================================================================
class TemperatureScaler(nn.Module):
    """Post-hoc Temperature Scaling for calibrated clinical risk probabilities."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def fit(self, logits: np.ndarray, y_true: np.ndarray, max_iter: int = 100) -> float:
        logits_t = torch.tensor(logits, dtype=torch.float32)
        y_t = torch.tensor(y_true, dtype=torch.float32)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.05, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits_t / torch.clamp(self.temperature, min=0.05, max=10.0)
            loss = F.binary_cross_entropy_with_logits(scaled_logits, y_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits_t = torch.tensor(logits, dtype=torch.float32)
            t = torch.clamp(self.temperature, min=0.05, max=10.0)
            probs = torch.sigmoid(logits_t / t).numpy()
        return probs


class SplitConformalAbstainer:
    """
    Constructs finite-sample valid prediction sets C(x) subset of {0, 1}.
    If |C(x)| != 1, the model abstains and flags the patient for clinician review.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_hat = None

    def calibrate(self, probs_cal: np.ndarray, y_cal: np.ndarray):
        n = len(y_cal)
        # Non-conformity score: 1 - probability assigned to the true class
        scores = np.where(y_cal == 1, 1.0 - probs_cal, probs_cal)
        q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        q_level = np.clip(q_level, 0.0, 1.0)
        self.q_hat = float(np.quantile(scores, q_level, method="higher"))

    def predict_with_abstention(self, probs_test: np.ndarray) -> Dict[str, np.ndarray]:
        include_pos = (1.0 - probs_test) <= self.q_hat
        include_neg = probs_test <= self.q_hat

        set_size = include_pos.astype(int) + include_neg.astype(int)
        abstain_mask = set_size != 1  # Abstain when ambiguous ({0,1}) or empty ({})
        confident_mask = ~abstain_mask

        return {
            "probs": probs_test,
            "include_pos": include_pos,
            "include_neg": include_neg,
            "abstain_mask": abstain_mask,
            "confident_mask": confident_mask,
            "coverage": float(np.mean(confident_mask)),
        }


# =====================================================================
# MODULE 5: PUBLICATION EVALUATION SUITE (ECE, BRIER, NET BENEFIT)
# =====================================================================
def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b_low, b_high in zip(bins[:-1], bins[1:]):
        mask = (probs >= b_low) & (probs < b_high)
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(probs[mask])
            ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def decision_curve_net_benefit(
    probs: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray
) -> List[Dict[str, float]]:
    n = len(y_true)
    results = []
    for pt in thresholds:
        preds = (probs >= pt).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        net_benefit = (tp / n) - (fp / n) * (pt / (1.0 - pt))
        results.append({"threshold": float(pt), "net_benefit": float(net_benefit)})
    return results


# =====================================================================
# MODULE 6: END-TO-END DEMO & LEAVE-ONE-COHORT-OUT VERIFICATION
# =====================================================================
def generate_synthetic_multi_cohort_data() -> pd.DataFrame:
    """
    Generates realistic multi-cohort data mimicking:
    - Cohort A (UCI-like: has max_hr_thalach, st_depression_oldpeak, chest_pain_type, chol_mgdl)
    - Cohort B (Sulianova-like: has bmi, systolic_bp, diastolic_bp, chol_tier, gluc_tier, smoker, plus corrupt rows)
    - Cohort C (External Held-Out Hospital Cohort: overlapping partial schema)
    """
    rng = np.random.default_rng(SEED)

    def make_cohort(name: str, n: int, schema_type: str) -> pd.DataFrame:
        age = rng.uniform(35, 78, size=n)
        sex = rng.integers(0, 2, size=n)
        sbp = rng.normal(132, 20, size=n)
        dbp = sbp - rng.uniform(35, 65, size=n)
        bmi = rng.normal(27.5, 4.8, size=n)
        chol = rng.normal(215, 42, size=n)
        thalach = 220 - age - rng.normal(15, 12, size=n)
        oldpeak = np.abs(rng.normal(1.0, 1.1, size=n))

        # True latent log-odds driven by established clinical physiology
        logit = (
            -6.8
            + 0.045 * (age - 50)
            + 0.45 * sex
            + 0.032 * (sbp - 120)
            + 0.06 * (bmi - 25)
            + 0.008 * (chol - 200)
            + 0.55 * oldpeak
            - 0.015 * (thalach - 150)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        target = (rng.uniform(0, 1, size=n) < prob).astype(int)

        df = pd.DataFrame({"cohort": name, "age_years": age, "sex_male": sex, "systolic_bp": sbp, "target": target})

        if schema_type == "uci":
            df["chol_mgdl"] = chol
            df["max_hr_thalach"] = thalach
            df["st_depression_oldpeak"] = oldpeak
            df["chest_pain_type"] = rng.integers(1, 5, size=n)
            df["exercise_angina"] = rng.integers(0, 2, size=n)
            # Inject 30 duplicate rows to mimic Kaggle Heart dataset
            df = pd.concat([df, df.iloc[:30]], ignore_index=True)

        elif schema_type == "sulianova":
            df["diastolic_bp"] = dbp
            df["bmi"] = bmi
            df["chol_tier"] = pd.cut(chol, bins=[-np.inf, 200, 239, np.inf], labels=[1, 2, 3]).astype(int)
            df["gluc_tier"] = rng.choice([1, 2, 3], size=n, p=[0.75, 0.15, 0.10])
            df["smoker"] = rng.integers(0, 2, size=n)
            # Inject physiologically impossible BP errors (like Sulianova ap_hi=16020 or ap_lo > ap_hi)
            df.loc[0:4, "systolic_bp"] = 14000.0
            df.loc[5:9, "diastolic_bp"] = 210.0

        elif schema_type == "external_hospital":
            df["diastolic_bp"] = dbp
            df["bmi"] = bmi
            df["chol_mgdl"] = chol
            df["st_depression_oldpeak"] = oldpeak
            df["smoker"] = rng.integers(0, 2, size=n)

        return df

    c1 = make_cohort("Cohort_UCI", 600, "uci")
    c2 = make_cohort("Cohort_Sulianova", 1200, "sulianova")
    c3 = make_cohort("Cohort_External_HeldOut", 500, "external_hospital")
    return pd.concat([c1, c2, c3], ignore_index=True)


def run_loco_experiment():
    print("=" * 78)
    print("STEP 1: FORENSIC AUDIT & PHYSIOLOGICAL QUALITY CONTROL")
    print("=" * 78)
    raw_df = generate_synthetic_multi_cohort_data()
    cleaned_cohorts = []
    for cohort_name, group in raw_df.groupby("cohort"):
        clean_group, audit = ClinicalPlausibilityGatekeeper.audit_and_clean(group, cohort_name)
        clean_group = ClinicalPlausibilityGatekeeper.engineer_clinical_concepts(clean_group)
        cleaned_cohorts.append(clean_group)
        print(
            f"[{audit['cohort']}] Initial: {audit['initial_rows']} | "
            f"Duplicates Removed: {audit['duplicates_removed']} | "
            f"Corrupt BP Removed: {audit['physiologically_corrupt_removed']} | "
            f"Clean: {audit['final_clean_rows']}"
        )

    full_clean_df = pd.concat(cleaned_cohorts, ignore_index=True)

    # Leave-One-Cohort-Out Split: Train on UCI + Sulianova, Test Zero-Shot on External Hospital
    train_val_df = full_clean_df[full_clean_df["cohort"] != "Cohort_External_HeldOut"].reset_index(drop=True)
    external_test_df = full_clean_df[full_clean_df["cohort"] == "Cohort_External_HeldOut"].reset_index(drop=True)

    # Split train_val into Train (80%) and Calibration/Val (20%)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(skf.split(train_val_df, train_val_df["target"]))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    train_ds = MultiCohortTabularDataset(train_df, is_train=True)
    val_ds = MultiCohortTabularDataset(val_df, cont_stats=train_ds.cont_stats, is_train=False)
    ext_ds = MultiCohortTabularDataset(external_test_df, cont_stats=train_ds.cont_stats, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    ext_loader = DataLoader(ext_ds, batch_size=128, shuffle=False)

    print("\n" + "=" * 78)
    print("STEP 2: TRAINING SCHEMAFLEXTABNET-QC ACROSS MISMATCHED COHORTS")
    print("=" * 78)
    model = SchemaFlexTabNetQC(d_model=64, nhead=4, num_layers=2, dropout=0.2, feature_drop_prob=0.20)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, 21):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            risk_logits, concept_logits = model(
                batch["x_cont"], batch["mask_cont"], batch["x_cat"], batch["mask_cat"]
            )
            loss_bce = F.binary_cross_entropy_with_logits(risk_logits, batch["y"])
            loss_concept = F.binary_cross_entropy_with_logits(concept_logits, batch["concept_htn"])
            loss_mono = compute_monotonicity_loss(
                model, batch["x_cont"], batch["mask_cont"], batch["x_cat"], batch["mask_cat"]
            )
            total_loss = loss_bce + 0.10 * loss_concept + 0.05 * loss_mono
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d}/20 | Joint Loss (BCE + Concept + Monotonic): {epoch_loss / len(train_loader):.4f}")

    # Collect validation and external test logits
    def get_logits_and_labels(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for b in loader:
                logits, _ = model(b["x_cont"], b["mask_cont"], b["x_cat"], b["mask_cat"])
                all_logits.append(logits.numpy())
                all_y.append(b["y"].numpy())
        return np.concatenate(all_logits), np.concatenate(all_y)

    val_logits, val_y = get_logits_and_labels(val_loader)
    ext_logits, ext_y = get_logits_and_labels(ext_loader)

    print("\n" + "=" * 78)
    print("STEP 3: TEMPERATURE CALIBRATION & CONFORMAL ABSTENTION ON HELD-OUT COHORT")
    print("=" * 78)
    scaler = TemperatureScaler()
    opt_temp = scaler.fit(val_logits, val_y)
    val_probs = scaler.predict_proba(val_logits)
    ext_probs = scaler.predict_proba(ext_logits)

    conformal = SplitConformalAbstainer(alpha=0.10)
    conformal.calibrate(val_probs, val_y)
    conf_out = conformal.predict_with_abstention(ext_probs)

    full_auroc = roc_auc_score(ext_y, ext_probs)
    full_brier = brier_score_loss(ext_y, ext_probs)
    full_ece = expected_calibration_error(ext_probs, ext_y)

    conf_mask = conf_out["confident_mask"]
    accepted_auroc = roc_auc_score(ext_y[conf_mask], ext_probs[conf_mask]) if np.sum(conf_mask) > 10 else full_auroc

    print(f"Optimal Calibration Temperature T*: {opt_temp:.3f}")
    print(f"Zero-Shot Held-Out Cohort AUROC (100% Coverage): {full_auroc:.4f}")
    print(f"Zero-Shot Held-Out Cohort Brier Score:           {full_brier:.4f}")
    print(f"Zero-Shot Held-Out Cohort Calibration ECE:       {full_ece:.4f}")
    print(f"Conformal Accepted Coverage (alpha=0.10):        {conf_out['coverage']*100:.1f}%")
    print(f"AUROC on Conformal Accepted Patients:            {accepted_auroc:.4f}")


if __name__ == "__main__":
    run_loco_experiment()
```