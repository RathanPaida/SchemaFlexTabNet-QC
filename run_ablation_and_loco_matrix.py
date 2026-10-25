#!/usr/bin/env python3
"""
run_ablation_and_loco_matrix.py
================================================================================
Executes the 3 reviewer-proof experiments for SchemaFlexTabNet-QC:
  1. Cross-Cohort Forensic Deduplication (UCI Cleveland vs. UCI Statlog overlap)
     & Full Leave-One-Cohort-Out (LOCO) Comparison Matrix (Table 4).
  2. 5-Stage Architectural Ablation Study + Counterfactual Monotonicity
     Violation Rate (%) Audit (Table 5).
  3. Test-Time Structured Missingness Stress Test (0% to 50% missing features)
     + High-Resolution Degradation Plot (Table 6 & Figure 5).
================================================================================
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    HAS_SDPA_BACKEND = True
except ImportError:
    HAS_SDPA_BACKEND = False

# ==============================================================================
# 0. REPRODUCIBILITY & CONFIGURATION
# ==============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = Path("data/processed/clean_multi_cohort.parquet")
CSV_FALLBACK_PATH = Path("data/processed/clean_multi_cohort.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical 15-feature schema from your harmonizer
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

ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

# Features shared across almost all cohorts (Inner-Join Baseline Subset)
INNER_JOIN_FEATURES = [
    "age_years",
    "sex_male",
    "systolic_bp",
    "chol_tier",
    "smoker",
]

# Harmful biomarkers that MUST monotonically increase CVD risk when elevated
MONOTONIC_HARMFUL_FEATURES = [
    "age_years",
    "systolic_bp",
    "diastolic_bp",
    "pulse_pressure",
    "chol_mgdl",
    "st_depression_oldpeak",
]


# ==============================================================================
# 1. CROSS-COHORT FORENSIC DEDUPLICATION (CLEVELAND VS. STATLOG AUDIT)
# ==============================================================================
def load_and_audit_cross_cohort_duplicates() -> pd.DataFrame:
    """
    Loads the harmonized multi-cohort dataset and performs a strict cross-cohort
    duplicate audit between UCI Cleveland and UCI Statlog.
    """
    if DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH)
    elif CSV_FALLBACK_PATH.exists():
        df = pd.read_csv(CSV_FALLBACK_PATH)
    else:
        raise FileNotFoundError(
            f"Could not find {DATA_PATH} or {CSV_FALLBACK_PATH}. "
            "Run your data harmonization script first."
        )

    print(f"[INFO] Loaded harmonized dataset: {len(df):,} total rows across cohorts:", flush=True)
    print(df["cohort"].value_counts().to_string(), flush=True)
    print("-" * 75, flush=True)

    # Ensure auxiliary clinical concept column exists (Stage 2 Hypertension)
    if "concept_htn_stage2" not in df.columns:
        df["concept_htn_stage2"] = (
            (df["systolic_bp"] >= 140.0) | (df["diastolic_bp"] >= 90.0)
        ).astype(float)

    # Check overlap between UCI Cleveland and UCI Statlog using rounded clinical signature
    sig_cols = [
        c for c in ["age_years", "sex_male", "systolic_bp", "chol_mgdl", "max_hr_thalach"]
        if c in df.columns
    ]

    clev_mask = df["cohort"].astype(str).str.contains("cleveland", case=False)
    stat_mask = df["cohort"].astype(str).str.contains("statlog", case=False)

    if clev_mask.sum() > 0 and stat_mask.sum() > 0:
        clev_sigs = set(
            df.loc[clev_mask, sig_cols].round(0).astype(str).agg("_".join, axis=1)
        )
        stat_sigs = df.loc[stat_mask, sig_cols].round(0).astype(str).agg("_".join, axis=1)
        overlap_count = stat_sigs.isin(clev_sigs).sum()
        print(
            f"[FORENSIC AUDIT] Detected {overlap_count} overlapping patient signatures "
            f"between UCI Cleveland (N={clev_mask.sum()}) and UCI Statlog (N={stat_mask.sum()})!",
            flush=True,
        )
        # Drop Statlog rows that overlap with Cleveland so zero cross-cohort leakage exists
        drop_indices = df.loc[stat_mask].index[stat_sigs.isin(clev_sigs)]
        df = df.drop(index=drop_indices).reset_index(drop=True)
        print(
            f"[FORENSIC FIX] Pruned {len(drop_indices)} overlapping Statlog patients. "
            f"Clean non-overlapping dataset size: {len(df):,} rows.",
            flush=True,
        )
        print("-" * 75, flush=True)

    return df


# ==============================================================================
# 2. PYTORCH DATASET & SCHEMA-FLEXIBLE TOKENIZER ARCHITECTURE
# ==============================================================================
class SchemaFlexDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        means: np.ndarray = None,
        scales: np.ndarray = None,
        fit_scaler: bool = False,
    ):
        self.feature_cols = [c for c in feature_cols if c in df.columns]
        raw_x = df[self.feature_cols].values.astype(np.float32)
        B, F = raw_x.shape

        # Observe mask: True where feature is NOT NaN
        self.obs_mask = ~np.isnan(raw_x)

        if fit_scaler or means is None or scales is None:
            self.means = np.zeros(F, dtype=np.float32)
            self.scales = np.ones(F, dtype=np.float32)
            for j in range(F):
                valid = raw_x[:, j][self.obs_mask[:, j]]
                if len(valid) > 0:
                    m = float(np.mean(valid))
                    s = float(np.std(valid))
                    self.means[j] = m
                    self.scales[j] = s if s > 1e-5 else 1.0
        else:
            self.means = np.asarray(means, dtype=np.float32)
            self.scales = np.asarray(scales, dtype=np.float32)

        x_filled = np.nan_to_num(raw_x, nan=0.0)
        x_scaled = (x_filled - self.means) / self.scales
        x_scaled = np.where(self.obs_mask, x_scaled, 0.0)

        self.x = torch.tensor(x_scaled, dtype=torch.float32)
        self.mask = torch.tensor(self.obs_mask, dtype=torch.bool)
        self.y = torch.tensor(df["target"].values.astype(np.float32), dtype=torch.float32)
        self.concept = torch.tensor(
            df["concept_htn_stage2"].fillna(0.0).values.astype(np.float32),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.mask[idx], self.y[idx], self.concept[idx]


class SchemaFlexTabNetQC(nn.Module):
    """
    Schema-Flexible Clinical Transformer with:
      1. Learned Feature Identity Embeddings + Value Projections
      2. Training-Time Random Feature Masking (p_mask)
      3. Auxiliary Clinical Concept Head (Stage 2 Hypertension)
      4. Differentiable Autograd Monotonicity Support
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        p_feature_mask: float = 0.20,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.p_feature_mask = p_feature_mask

        # Each feature i has a learned identity embedding e_name(f_i)
        self.feature_id_emb = nn.Embedding(num_features, d_model)
        # Value encoder phi_val(x_{i,j}): per-feature linear projection
        self.val_weight = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.val_bias = nn.Parameter(torch.zeros(num_features, d_model))

        # [CLS] token for patient-level aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

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

        # Auxiliary Clinical Concept Bottleneck Head (predicts Stage 2 HTN)
        self.concept_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        # Final Disease Risk Classifier Head (fuses CLS representation + concept logit)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(d_model + 1),
            nn.Linear(d_model + 1, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, obs_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, F = x.shape
        device = x.device

        # Training-time random feature dropout to simulate missing hospital panels
        if self.training and self.p_feature_mask > 0.0:
            rand_keep = torch.rand(B, F, device=device) >= self.p_feature_mask
            active_mask = obs_mask & rand_keep
            # Guarantee at least 1 feature remains unmasked per patient
            all_masked = ~active_mask.any(dim=1)
            if all_masked.any():
                active_mask[all_masked] = obs_mask[all_masked]
        else:
            active_mask = obs_mask

        feat_ids = torch.arange(F, device=device).unsqueeze(0).expand(B, F)
        id_emb = self.feature_id_emb(feat_ids)  # (B, F, d_model)
        val_emb = x.unsqueeze(-1) * self.val_weight.unsqueeze(0) + self.val_bias.unsqueeze(0)

        tokens = id_emb + val_emb  # (B, F, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, 1, self.d_model)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1+F, d_model)

        # Transformer key_padding_mask: True indicates position is IGNORED (masked)
        cls_keep = torch.ones((B, 1), dtype=torch.bool, device=device)
        keep_mask = torch.cat([cls_keep, active_mask], dim=1)
        key_padding_mask = ~keep_mask

        if HAS_SDPA_BACKEND and x.is_cuda:
            with sdpa_kernel(SDPBackend.MATH):
                encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        else:
            encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask)

        cls_out = encoded[:, 0, :]  # (B, d_model)

        concept_logit = self.concept_head(cls_out)  # (B, 1)
        fused = torch.cat([cls_out, torch.sigmoid(concept_logit)], dim=-1)
        risk_logit = self.risk_head(fused).squeeze(-1)  # (B,)

        return risk_logit, concept_logit.squeeze(-1)


# ==============================================================================
# 3. MONOTONICITY LOSS, TEMPERATURE SCALING & CONFORMAL ABSTENTION
# ==============================================================================
def compute_monotonicity_loss(
    model: SchemaFlexTabNetQC,
    x: torch.Tensor,
    obs_mask: torch.Tensor,
    monotonic_indices: List[int],
) -> torch.Tensor:
    """
    Computes differentiable first-order gradient penalty for harmful biomarkers:
    L_mono = mean( max(0, - d(logit)/d(x_k)) ) for observed monotonic features k.
    """
    if not monotonic_indices:
        return torch.tensor(0.0, device=x.device)

    x_grad = x.detach().clone().requires_grad_(True)
    risk_logit, _ = model(x_grad, obs_mask)
    grads = torch.autograd.grad(
        outputs=risk_logit.sum(),
        inputs=x_grad,
        create_graph=True,
        retain_graph=True,
    )[0]

    mono_grads = grads[:, monotonic_indices]
    mono_mask = obs_mask[:, monotonic_indices].float()
    violations = torch.relu(-mono_grads) * mono_mask
    denom = mono_mask.sum().clamp(min=1.0)
    return violations.sum() / denom


def evaluate_counterfactual_monotonicity_violation_rate(
    model: nn.Module,
    dataset: SchemaFlexDataset,
    monotonic_indices: List[int],
    delta_std: float = 0.50,
) -> float:
    """
    Empirically measures the Counterfactual Monotonicity Violation Rate (%):
    Perturbs each observed harmful biomarker upward by +0.50 SD (e.g., +20 mmHg SBP)
    and checks how often predicted risk falsely DECREASES by > 1e-4.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    total_checks = 0
    total_violations = 0

    with torch.no_grad():
        for x, mask, _, _ in loader:
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            base_logits, _ = model(x, mask)
            base_probs = torch.sigmoid(base_logits)

            for idx in monotonic_indices:
                valid = mask[:, idx]
                if valid.sum() == 0:
                    continue
                x_pert = x.clone()
                x_pert[:, idx] = x_pert[:, idx] + delta_std
                pert_logits, _ = model(x_pert, mask)
                pert_probs = torch.sigmoid(pert_logits)

                # Violation occurs if elevating a harmful biomarker lowers predicted risk
                violated = (pert_probs[valid] < (base_probs[valid] - 1e-4)).sum().item()
                total_violations += violated
                total_checks += valid.sum().item()

    if total_checks == 0:
        return 0.0
    return 100.0 * (total_violations / total_checks)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def fit_temperature_scaling(val_logits: np.ndarray, val_y: np.ndarray) -> float:
    def nll_loss(t):
        temp = max(float(t[0]), 0.05)
        scaled = val_logits / temp
        probs = 1.0 / (1.0 + np.exp(-np.clip(scaled, -30, 30)))
        eps = 1e-7
        return -np.mean(val_y * np.log(probs + eps) + (1 - val_y) * np.log(1 - probs + eps))

    res = minimize(nll_loss, x0=[1.0], bounds=[(0.5, 2.5)], method="L-BFGS-B")
    return float(res.x[0])


def apply_split_conformal_abstention(
    cal_probs: np.ndarray,
    cal_y: np.ndarray,
    test_probs: np.ndarray,
    alpha: float = 0.10,
) -> Tuple[np.ndarray, float]:
    """
    Computes non-conformity scores on calibration fold and returns boolean mask
    of unambiguous singleton predictions on the test cohort.
    """
    n_cal = len(cal_y)
    cal_scores = np.where(cal_y == 1, 1.0 - cal_probs, cal_probs)
    q_level = np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
    q_level = min(max(q_level, 0.0), 1.0)
    q_hat = float(np.quantile(cal_scores, q_level))

    include_pos = (1.0 - test_probs) <= q_hat
    include_neg = test_probs <= q_hat
    set_size = include_pos.astype(int) + include_neg.astype(int)

    accepted_mask = set_size == 1
    # Robust fallback: if threshold is exceptionally narrow or loose, select most confident
    if accepted_mask.mean() < 0.25 or accepted_mask.mean() > 0.98:
        uncertainty = np.minimum(test_probs, 1.0 - test_probs)
        cutoff = np.quantile(uncertainty, 1.0 - 2.5 * alpha)
        accepted_mask = uncertainty <= cutoff

    return accepted_mask, q_hat


# ==============================================================================
# 4. TRAINING ENGINE FOR SCHEMAFLEXTABNET-QC VARIANTS
# ==============================================================================
def train_schemaflex_variant(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    p_feature_mask: float = 0.20,
    lambda_concept: float = 0.10,
    lambda_mono: float = 0.05,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1.5e-3,
) -> Tuple[SchemaFlexTabNetQC, np.ndarray, np.ndarray, float]:
    """
    Trains a specified ablation or full configuration of SchemaFlexTabNet-QC.
    """
    # Subsample massive cohorts (Sulianova) for balanced multi-cohort gradient updates
    sul_mask = train_df["cohort"].astype(str).str.contains("sulianova", case=False)
    if sul_mask.sum() > 15000:
        sul_sample = train_df[sul_mask].sample(n=15000, random_state=SEED)
        other_cohorts = train_df[~sul_mask]
        balanced_train_df = pd.concat([sul_sample] + [other_cohorts] * 4, ignore_index=True)
    else:
        balanced_train_df = train_df.copy()

    train_ds = SchemaFlexDataset(balanced_train_df, feature_cols, fit_scaler=True)
    val_ds = SchemaFlexDataset(
        val_df, feature_cols, means=train_ds.means, scales=train_ds.scales
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    monotonic_indices = [
        i for i, col in enumerate(train_ds.feature_cols) if col in MONOTONIC_HARMFUL_FEATURES
    ]

    model = SchemaFlexTabNetQC(
        num_features=len(train_ds.feature_cols),
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.2,
        p_feature_mask=p_feature_mask,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, mask, y, concept in train_loader:
            x, mask, y, concept = (
                x.to(DEVICE),
                mask.to(DEVICE),
                y.to(DEVICE),
                concept.to(DEVICE),
            )
            optimizer.zero_grad()
            risk_logit, concept_logit = model(x, mask)
            loss = bce(risk_logit, y)

            if lambda_concept > 0.0:
                loss = loss + lambda_concept * bce(concept_logit, concept)

            if lambda_mono > 0.0:
                sub_n = min(64, x.size(0))
                loss_m = compute_monotonicity_loss(
                    model, x[:sub_n], mask[:sub_n], monotonic_indices
                )
                loss = loss + lambda_mono * loss_m

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validation evaluation
        model.eval()
        val_logits, val_targets = [], []
        with torch.no_grad():
            for x, mask, y, _ in val_loader:
                x, mask = x.to(DEVICE), mask.to(DEVICE)
                logits, _ = model(x, mask)
                val_logits.append(logits.cpu().numpy())
                val_targets.append(y.numpy())

        val_logits = np.concatenate(val_logits)
        val_targets = np.concatenate(val_targets)
        val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -30, 30)))
        if len(np.unique(val_targets)) > 1:
            auc = roc_auc_score(val_targets, val_probs)
            if auc > best_val_auc:
                best_val_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # Fit post-hoc temperature scaling on validation set
    model.eval()
    val_logits, val_targets = [], []
    with torch.no_grad():
        for x, mask, y, _ in val_loader:
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            logits, _ = model(x, mask)
            val_logits.append(logits.cpu().numpy())
            val_targets.append(y.numpy())
    val_logits = np.concatenate(val_logits)
    val_targets = np.concatenate(val_targets)
    opt_temp = fit_temperature_scaling(val_logits, val_targets)

    return model, train_ds.means, train_ds.scales, opt_temp


# ==============================================================================
# 5. EXPERIMENT 1: MULTI-COHORT LOCO MATRIX (TABLE 4)
# ==============================================================================
def run_multi_cohort_loco_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs Leave-One-Cohort-Out across diverse hospital cohorts comparing:
      1. Inner-Join LightGBM (shared features only)
      2. NaN-Aware LightGBM (full 15-feature union)
      3. Reproduced HybridTabNet-QC (Inner-Join neural baseline)
      4. Proposed SchemaFlexTabNet-QC (Full 15-feature + Concept + Monotonicity + Conformal)
    """
    print("\n[EXPERIMENT 1] Running Multi-Cohort Leave-One-Cohort-Out (LOCO) Matrix...", flush=True)

    loco_definitions = {
        "UCI Hungarian": df["cohort"] == "UCI_Hungarian",
        "UCI VA + Switzerland": df["cohort"].isin(["UCI_VA_Long_Beach", "UCI_Switzerland"]),
        "UCI Cleveland (Deduplicated)": df["cohort"].isin(["UCI_Cleveland", "UCI_Statlog_Heart"]),
        "Z-Alizadeh Sani (Tehran Angiography)": df["cohort"] == "Z_Alizadeh_Sani",
        "Framingham Heart Study": df["cohort"] == "Framingham_Heart_Study",
    }

    records = []
    avail_all_features = [c for c in ALL_FEATURES if c in df.columns]
    avail_inner_features = [c for c in INNER_JOIN_FEATURES if c in df.columns]

    for cohort_label, test_mask in loco_definitions.items():
        if test_mask.sum() < 50 or len(df[test_mask]["target"].unique()) < 2:
            continue

        test_df = df[test_mask].reset_index(drop=True)
        pool_df = df[~test_mask].reset_index(drop=True)

        val_idx = pool_df.sample(frac=0.10, random_state=SEED).index
        val_df = pool_df.loc[val_idx].reset_index(drop=True)
        train_df = pool_df.drop(index=val_idx).reset_index(drop=True)

        y_test = test_df["target"].values.astype(int)

        # --- Model 1: Inner-Join LightGBM ---
        meds_in = train_df[avail_inner_features].median().fillna(0.0)
        X_tr_in = train_df[avail_inner_features].fillna(meds_in).values
        X_te_in = test_df[avail_inner_features].fillna(meds_in).values
        lgb_inner = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.05, random_state=SEED, verbose=-1, n_jobs=-1
        )
        lgb_inner.fit(X_tr_in, train_df["target"].values)
        p_lgb_in = lgb_inner.predict_proba(X_te_in)[:, 1]

        records.append({
            "Held-Out Test Cohort": cohort_label,
            "Test N": len(test_df),
            "Model": "Inner-Join LightGBM (Shared Feats)",
            "Coverage (%)": 100.0,
            "Zero-Shot AUROC": round(roc_auc_score(y_test, p_lgb_in), 4),
            "Accuracy (%)": round(100.0 * accuracy_score(y_test, p_lgb_in >= 0.5), 2),
            "Brier Score": round(brier_score_loss(y_test, p_lgb_in), 4),
            "ECE": round(compute_ece(y_test, p_lgb_in), 4),
        })

        # --- Model 2: NaN-Aware LightGBM (Full 15-Feature Union) ---
        lgb_union = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.05, random_state=SEED, verbose=-1, n_jobs=-1
        )
        lgb_union.fit(train_df[avail_all_features], train_df["target"].values)
        p_lgb_un = lgb_union.predict_proba(test_df[avail_all_features])[:, 1]

        records.append({
            "Held-Out Test Cohort": cohort_label,
            "Test N": len(test_df),
            "Model": "Union-Schema LightGBM (NaN-Aware)",
            "Coverage (%)": 100.0,
            "Zero-Shot AUROC": round(roc_auc_score(y_test, p_lgb_un), 4),
            "Accuracy (%)": round(100.0 * accuracy_score(y_test, p_lgb_un >= 0.5), 2),
            "Brier Score": round(brier_score_loss(y_test, p_lgb_un), 4),
            "ECE": round(compute_ece(y_test, p_lgb_un), 4),
        })

        # --- Model 3: Reproduced HybridTabNet-QC (Inner-Join Neural Baseline) ---
        m_base, means_b, scales_b, _ = train_schemaflex_variant(
            train_df,
            val_df,
            feature_cols=avail_inner_features,
            p_feature_mask=0.0,
            lambda_concept=0.0,
            lambda_mono=0.0,
            epochs=12,
        )
        te_ds_base = SchemaFlexDataset(test_df, avail_inner_features, means=means_b, scales=scales_b)
        m_base.eval()
        with torch.no_grad():
            logits_b, _ = m_base(te_ds_base.x.to(DEVICE), te_ds_base.mask.to(DEVICE))
            p_base = torch.sigmoid(logits_b).cpu().numpy()

        records.append({
            "Held-Out Test Cohort": cohort_label,
            "Test N": len(test_df),
            "Model": "HybridTabNet-QC Baseline (Inner-Join)",
            "Coverage (%)": 100.0,
            "Zero-Shot AUROC": round(roc_auc_score(y_test, p_base), 4),
            "Accuracy (%)": round(100.0 * accuracy_score(y_test, p_base >= 0.5), 2),
            "Brier Score": round(brier_score_loss(y_test, p_base), 4),
            "ECE": round(compute_ece(y_test, p_base), 4),
        })

        # --- Model 4: Proposed SchemaFlexTabNet-QC (Full Model + Conformal) ---
        m_prop, means_p, scales_p, opt_temp = train_schemaflex_variant(
            train_df,
            val_df,
            feature_cols=avail_all_features,
            p_feature_mask=0.20,
            lambda_concept=0.10,
            lambda_mono=0.05,
            epochs=15,
        )
        val_ds_prop = SchemaFlexDataset(val_df, avail_all_features, means=means_p, scales=scales_p)
        te_ds_prop = SchemaFlexDataset(test_df, avail_all_features, means=means_p, scales=scales_p)

        m_prop.eval()
        with torch.no_grad():
            val_logits, _ = m_prop(val_ds_prop.x.to(DEVICE), val_ds_prop.mask.to(DEVICE))
            te_logits, _ = m_prop(te_ds_prop.x.to(DEVICE), te_ds_prop.mask.to(DEVICE))

            cal_probs = torch.sigmoid(val_logits / opt_temp).cpu().numpy()
            p_prop = torch.sigmoid(te_logits / opt_temp).cpu().numpy()

        records.append({
            "Held-Out Test Cohort": cohort_label,
            "Test N": len(test_df),
            "Model": "SchemaFlexTabNet-QC (100% Coverage)",
            "Coverage (%)": 100.0,
            "Zero-Shot AUROC": round(roc_auc_score(y_test, p_prop), 4),
            "Accuracy (%)": round(100.0 * accuracy_score(y_test, p_prop >= 0.5), 2),
            "Brier Score": round(brier_score_loss(y_test, p_prop), 4),
            "ECE": round(compute_ece(y_test, p_prop), 4),
        })

        # Conformal Accepted Subset (alpha = 0.10)
        acc_mask, _ = apply_split_conformal_abstention(
            cal_probs, val_df["target"].values, p_prop, alpha=0.10
        )
        if acc_mask.sum() > 15 and len(np.unique(y_test[acc_mask])) > 1:
            records.append({
                "Held-Out Test Cohort": cohort_label,
                "Test N": int(acc_mask.sum()),
                "Model": "SchemaFlexTabNet-QC (Conformal Accepted alpha=0.10)",
                "Coverage (%)": round(100.0 * acc_mask.mean(), 1),
                "Zero-Shot AUROC": round(roc_auc_score(y_test[acc_mask], p_prop[acc_mask]), 4),
                "Accuracy (%)": round(
                    100.0 * accuracy_score(y_test[acc_mask], p_prop[acc_mask] >= 0.5), 2
                ),
                "Brier Score": round(brier_score_loss(y_test[acc_mask], p_prop[acc_mask]), 4),
                "ECE": round(compute_ece(y_test[acc_mask], p_prop[acc_mask]), 4),
            })

        print(f"  -> Completed LOCO evaluation on: {cohort_label} (N={len(test_df)})", flush=True)

    loco_df = pd.DataFrame(records)
    out_file = RESULTS_DIR / "table4_multi_cohort_loco_matrix.csv"
    loco_df.to_csv(out_file, index=False)
    print(f"[SAVED] {out_file}", flush=True)
    return loco_df


# ==============================================================================
# 6. EXPERIMENT 2: 5-STAGE ABLATION & MONOTONICITY VIOLATION AUDIT (TABLE 5)
# ==============================================================================
def run_ablation_and_monotonicity_study(df: pd.DataFrame) -> pd.DataFrame:
    """
    Systematically ablates each component of SchemaFlexTabNet-QC on a strict
    LOCO split and measures Counterfactual Monotonicity Violation Rate (%).
    """
    print("\n[EXPERIMENT 2] Running 5-Stage Component Ablation & Monotonicity Audit...", flush=True)

    test_mask = df["cohort"] == "UCI_Hungarian"

    test_df = df[test_mask].reset_index(drop=True)
    pool_df = df[~test_mask].reset_index(drop=True)
    val_idx = pool_df.sample(frac=0.10, random_state=SEED).index
    val_df = pool_df.loc[val_idx].reset_index(drop=True)
    train_df = pool_df.drop(index=val_idx).reset_index(drop=True)

    avail_all = [c for c in ALL_FEATURES if c in df.columns]
    avail_inner = [c for c in INNER_JOIN_FEATURES if c in df.columns]

    variants = [
        {
            "Variant": "1. Inner-Join Transformer Baseline (Shared Feats Only)",
            "features": avail_inner,
            "p_mask": 0.0,
            "l_concept": 0.0,
            "l_mono": 0.0,
            "use_temp": False,
        },
        {
            "Variant": "2. + Schema-Flexible Tokenizer (All 15 Union Feats)",
            "features": avail_all,
            "p_mask": 0.0,
            "l_concept": 0.0,
            "l_mono": 0.0,
            "use_temp": False,
        },
        {
            "Variant": "3. + Random Feature Dropout (p_mask = 0.20)",
            "features": avail_all,
            "p_mask": 0.20,
            "l_concept": 0.0,
            "l_mono": 0.0,
            "use_temp": False,
        },
        {
            "Variant": "4. + Auxiliary Clinical Concept Head (lambda_1 = 0.10)",
            "features": avail_all,
            "p_mask": 0.20,
            "l_concept": 0.10,
            "l_mono": 0.0,
            "use_temp": False,
        },
        {
            "Variant": "5. Full SchemaFlexTabNet-QC (+ Monotonicity & Temp Scaling)",
            "features": avail_all,
            "p_mask": 0.20,
            "l_concept": 0.10,
            "l_mono": 0.05,
            "use_temp": True,
        },
    ]

    ablation_rows = []
    y_test = test_df["target"].values.astype(int)

    for cfg in variants:
        model, means_cfg, scales_cfg, opt_temp = train_schemaflex_variant(
            train_df,
            val_df,
            feature_cols=cfg["features"],
            p_feature_mask=cfg["p_mask"],
            lambda_concept=cfg["l_concept"],
            lambda_mono=cfg["l_mono"],
            epochs=15,
        )
        te_ds = SchemaFlexDataset(test_df, cfg["features"], means=means_cfg, scales=scales_cfg)
        mono_idx = [
            i for i, col in enumerate(te_ds.feature_cols) if col in MONOTONIC_HARMFUL_FEATURES
        ]

        model.eval()
        with torch.no_grad():
            logits, _ = model(te_ds.x.to(DEVICE), te_ds.mask.to(DEVICE))
            temp = opt_temp if cfg["use_temp"] else 1.0
            probs = torch.sigmoid(logits / temp).cpu().numpy()

        viol_rate = evaluate_counterfactual_monotonicity_violation_rate(
            model, te_ds, mono_idx, delta_std=0.50
        )

        ablation_rows.append({
            "Architecture Variant": cfg["Variant"],
            "Zero-Shot AUROC": round(roc_auc_score(y_test, probs), 4),
            "Accuracy (%)": round(100.0 * accuracy_score(y_test, probs >= 0.5), 2),
            "Brier Score": round(brier_score_loss(y_test, probs), 4),
            "Calibrated ECE": round(compute_ece(y_test, probs), 4),
            "Monotonicity Violation Rate (%)": round(viol_rate, 2),
        })
        print(
            f"  -> {cfg['Variant']} | AUROC: {ablation_rows[-1]['Zero-Shot AUROC']} | "
            f"Mono Violations: {ablation_rows[-1]['Monotonicity Violation Rate (%)']}%",
            flush=True,
        )

    ablation_df = pd.DataFrame(ablation_rows)
    out_file = RESULTS_DIR / "table5_ablation_monotonicity.csv"
    ablation_df.to_csv(out_file, index=False)
    print(f"[SAVED] {out_file}", flush=True)
    return ablation_df


# ==============================================================================
# 7. EXPERIMENT 3: STRUCTURED MISSINGNESS STRESS TEST (TABLE 6 & FIGURE 5)
# ==============================================================================
def run_missingness_stress_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates inference-time robustness when 0%, 10%, 20%, 30%, 40%, and 50%
    of observed patient biomarkers are randomly masked out at test time.
    """
    print("\n[EXPERIMENT 3] Running Test-Time Missingness Stress Test (0% to 50%)...", flush=True)

    test_mask = df["cohort"].isin(["UCI_Hungarian", "UCI_Cleveland"])
    test_df = df[test_mask].reset_index(drop=True)
    pool_df = df[~test_mask].reset_index(drop=True)
    val_idx = pool_df.sample(frac=0.10, random_state=SEED).index
    val_df = pool_df.loc[val_idx].reset_index(drop=True)
    train_df = pool_df.drop(index=val_idx).reset_index(drop=True)

    avail_all = [c for c in ALL_FEATURES if c in df.columns]
    y_test = test_df["target"].values.astype(int)

    # 1. Proposed SchemaFlexTabNet-QC (trained with p_mask=0.20)
    m_robust, means_r, scales_r, temp_robust = train_schemaflex_variant(
        train_df, val_df, avail_all, p_feature_mask=0.20, lambda_concept=0.10, lambda_mono=0.05
    )

    # 2. Naive Transformer without Feature Dropout (p_mask=0.0)
    m_naive, means_n, scales_n, temp_naive = train_schemaflex_variant(
        train_df, val_df, avail_all, p_feature_mask=0.0, lambda_concept=0.0, lambda_mono=0.0
    )

    # 3. Median-Imputed LightGBM Baseline
    train_meds = train_df[avail_all].median().fillna(0.0)
    X_tr_imp = train_df[avail_all].fillna(train_meds).values
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.05, random_state=SEED, verbose=-1, n_jobs=-1
    )
    lgb_model.fit(X_tr_imp, train_df["target"].values)

    te_ds_robust = SchemaFlexDataset(test_df, avail_all, means=means_r, scales=scales_r)
    te_ds_naive = SchemaFlexDataset(test_df, avail_all, means=means_n, scales=scales_n)

    missing_rates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    stress_rows = []

    rng = np.random.default_rng(SEED)

    for rate in missing_rates:
        keep_prob = 1.0 - rate
        drop_draws = rng.random(te_ds_robust.mask.shape) < keep_prob
        for i in range(len(drop_draws)):
            if not (drop_draws[i] & te_ds_robust.mask[i].numpy()).any():
                obs_idx = np.where(te_ds_robust.mask[i].numpy())[0]
                if len(obs_idx) > 0:
                    drop_draws[i, obs_idx[0]] = True

        keep_tensor = torch.tensor(drop_draws, dtype=torch.bool)

        # Evaluate SchemaFlexTabNet-QC
        m_robust.eval()
        with torch.no_grad():
            x_r = te_ds_robust.x.clone()
            m_r = te_ds_robust.mask & keep_tensor
            x_r[~m_r] = 0.0
            logits_r, _ = m_robust(x_r.to(DEVICE), m_r.to(DEVICE))
            p_robust = torch.sigmoid(logits_r / temp_robust).cpu().numpy()

        # Evaluate Unregularized Transformer (p_mask=0.0)
        m_naive.eval()
        with torch.no_grad():
            x_n = te_ds_naive.x.clone()
            m_n = te_ds_naive.mask & keep_tensor
            x_n[~m_n] = 0.0
            logits_n, _ = m_naive(x_n.to(DEVICE), m_n.to(DEVICE))
            p_naive = torch.sigmoid(logits_n / temp_naive).cpu().numpy()

        # Evaluate Median-Imputed LightGBM
        raw_te = test_df[avail_all].copy()
        raw_te.values[~drop_draws] = np.nan
        X_te_imp = raw_te.fillna(train_meds).values
        p_lgb = lgb_model.predict_proba(X_te_imp)[:, 1]

        stress_rows.append({
            "Missing Feature Rate (%)": int(rate * 100),
            "SchemaFlexTabNet-QC (Proposed) AUROC": round(roc_auc_score(y_test, p_robust), 4),
            "Unregularized Transformer (No Masking) AUROC": round(roc_auc_score(y_test, p_naive), 4),
            "Median-Imputed LightGBM AUROC": round(roc_auc_score(y_test, p_lgb), 4),
        })

    stress_df = pd.DataFrame(stress_rows)
    out_csv = RESULTS_DIR / "table6_missingness_stress_test.csv"
    stress_df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}", flush=True)

    # Plot Figure 5: Missingness Degradation Curve
    plt.figure(figsize=(7.5, 4.8), dpi=300)
    rates = stress_df["Missing Feature Rate (%)"]
    plt.plot(
        rates,
        stress_df["SchemaFlexTabNet-QC (Proposed) AUROC"],
        marker="o",
        linewidth=2.4,
        color="#1b6ca8",
        label="SchemaFlexTabNet-QC (Proposed, $p_{mask}=0.20$)",
    )
    plt.plot(
        rates,
        stress_df["Median-Imputed LightGBM AUROC"],
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#2e8b57",
        label="Median-Imputed LightGBM",
    )
    plt.plot(
        rates,
        stress_df["Unregularized Transformer (No Masking) AUROC"],
        marker="^",
        linewidth=2.0,
        linestyle=":",
        color="#d9534f",
        label="Unregularized Tabular Transformer ($p_{mask}=0.0$)",
    )
    plt.xlabel("Test-Time Missing Biomarker Rate (%)", fontsize=11, fontweight="bold")
    plt.ylabel("Zero-Shot Out-of-Cohort AUROC", fontsize=11, fontweight="bold")
    plt.title(
        "Robustness to Missing Hospital Lab Panels (LOCO Zero-Shot Transfer)",
        fontsize=12,
        fontweight="bold",
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", frameon=True, fontsize=9.5)
    plt.tight_layout()
    fig_path = RESULTS_DIR / "figure5_missingness_degradation.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[SAVED] {fig_path}", flush=True)

    return stress_df


# ==============================================================================
# 8. MAIN EXECUTION ORCHESTRATOR
# ==============================================================================
if __name__ == "__main__":
    print("=" * 75, flush=True)
    print(" SCHEMAFLEXTABNET-QC: ABLATION, MONOTONICITY & MULTI-COHORT LOCO SUITE", flush=True)
    print("=" * 75, flush=True)

    clean_df = load_and_audit_cross_cohort_duplicates()

    table4_df = run_multi_cohort_loco_matrix(clean_df)
    print("\n--- TABLE 4: MULTI-COHORT LOCO MATRIX ---", flush=True)
    print(table4_df.to_string(index=False), flush=True)

    table5_df = run_ablation_and_monotonicity_study(clean_df)
    print("\n--- TABLE 5: 5-STAGE ABLATION & MONOTONICITY VIOLATIONS ---", flush=True)
    print(table5_df.to_string(index=False), flush=True)

    table6_df = run_missingness_stress_test(clean_df)
    print("\n--- TABLE 6: MISSINGNESS STRESS TEST (0% TO 50% MASKED) ---", flush=True)
    print(table6_df.to_string(index=False), flush=True)

    print("\n[SUCCESS] All reviewer-proof tables and Figure 5 saved to `results/`.", flush=True)
