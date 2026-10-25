"""
Phase 4 & 5: Leave-One-Cohort-Out (LOCO) Pipeline for SchemaFlexTabNet-QC
Executes:
1. Multi-Cohort Training without Inner Join (Masked Self-Attention + Random Feature Masking)
2. Auxiliary Concept Bottleneck Loss + Differentiable Monotonicity Penalty
3. Post-Hoc Temperature Scaling for Probability Calibration
4. Split Conformal Prediction with Clinician Abstention (Risk-Coverage Curve)
5. 95% Bootstrap Confidence Intervals (1,000 resamples)
6. Subgroup Fairness Audit across Sex and Age Bands (Equalized Odds Disparity)
7. Decision Curve Analysis (DCA Net Benefit)
Produces:
- results/table3_proposed_loco_conformal.csv
- results/fairness_subgroup_audit.csv
- results/decision_curve_analysis.csv
- results/risk_coverage_curve.csv
"""

import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

from src.data_harmonizer import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
from src.schema_tokenizer import MultiCohortTabularDataset
from src.model import SchemaFlexTabNetQC, compute_monotonicity_loss
from src.conformal_qc import TemperatureScaler, SplitConformalAbstainer
from src.evaluate_metrics import (
    compute_all_metrics,
    expected_calibration_error,
    bootstrap_confidence_intervals,
    decision_curve_net_benefit,
    subgroup_fairness_audit,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logits_and_labels(
    model: SchemaFlexTabNetQC, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts raw model risk logits, concept logits, and ground-truth targets."""
    model.eval()
    all_logits, all_concepts, all_y = [], [], []
    with torch.no_grad():
        for b in loader:
            x_cont = b["x_cont"].to(device)
            mask_cont = b["mask_cont"].to(device)
            x_cat = b["x_cat"].to(device)
            mask_cat = b["mask_cat"].to(device)
            logits, concepts = model(x_cont, mask_cont, x_cat, mask_cat)
            all_logits.append(logits.cpu().numpy())
            all_concepts.append(concepts.cpu().numpy())
            all_y.append(b["y"].numpy())
    return np.concatenate(all_logits), np.concatenate(all_concepts), np.concatenate(all_y)


def train_loco_pipeline(
    held_out_cohort: str = "UCI_Statlog_Heart",
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1.5e-3,
    alpha: float = 0.10,
) -> Dict[str, pd.DataFrame]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"LEAVE-ONE-COHORT-OUT PIPELINE: SCHEMAFLEXTABNET-QC (Device: {DEVICE})", flush=True)
    print(f"Held-Out Zero-Shot Evaluation Cohort: {held_out_cohort}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Harmonized Multi-Cohort Dataset
    df_path = os.path.join(processed_dir, "clean_multi_cohort.parquet")
    full_df = pd.read_parquet(df_path)

    # Balance large cohorts (e.g. Sulianova) to ensure balanced representation
    balanced_subsets = []
    for cname, grp in full_df.groupby("cohort"):
        if cname == "Sulianova_CVD" and len(grp) > 15000:
            balanced_subsets.append(grp.sample(n=15000, random_state=SEED))
        else:
            balanced_subsets.append(grp)
    df = pd.concat(balanced_subsets, ignore_index=True)

    # 2. Leave-One-Cohort-Out Partition
    train_val_df = df[df["cohort"] != held_out_cohort].reset_index(drop=True)
    test_df = df[df["cohort"] == held_out_cohort].reset_index(drop=True)

    print(f"Training Pool: {len(train_val_df):,} samples across cohorts: {list(train_val_df['cohort'].unique())}", flush=True)
    print(f"Zero-Shot Test Set: {len(test_df):,} samples ({held_out_cohort})", flush=True)

    # 80/20 Train vs Calibration/Validation split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(skf.split(train_val_df, train_val_df["target"]))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # Datasets and Loaders
    train_ds = MultiCohortTabularDataset(train_df, is_train=True)
    val_ds = MultiCohortTabularDataset(val_df, cont_stats=train_ds.cont_stats, is_train=False)
    test_ds = MultiCohortTabularDataset(test_df, cont_stats=train_ds.cont_stats, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Initialize Model and Optimizer
    model = SchemaFlexTabNetQC(
        num_cont=len(CONTINUOUS_FEATURES),
        num_cat=len(CATEGORICAL_FEATURES),
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.20,
        feature_drop_prob=0.20,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 4. Joint Training Loop
    print("\nTraining SchemaFlexTabNet-QC (BCE + Concept Loss + Monotonicity Penalty)...", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_accum = 0.0
        bce_accum = 0.0
        concept_accum = 0.0
        mono_accum = 0.0

        for b_idx, batch in enumerate(train_loader):
            x_cont = batch["x_cont"].to(DEVICE)
            mask_cont = batch["mask_cont"].to(DEVICE)
            x_cat = batch["x_cat"].to(DEVICE)
            mask_cat = batch["mask_cat"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            concept_htn = batch["concept_htn"].to(DEVICE)

            optimizer.zero_grad()
            risk_logits, concept_logits = model(x_cont, mask_cont, x_cat, mask_cat)

            loss_bce = F.binary_cross_entropy_with_logits(risk_logits, y)
            loss_concept = F.binary_cross_entropy_with_logits(concept_logits, concept_htn)

            # Efficient monotonicity loss computation on first 64 samples of batch
            sub_n = min(64, x_cont.size(0))
            loss_mono = compute_monotonicity_loss(
                model,
                x_cont[:sub_n],
                mask_cont[:sub_n],
                x_cat[:sub_n],
                mask_cat[:sub_n],
            )

            loss_total = loss_bce + 0.10 * loss_concept + 0.05 * loss_mono
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss_accum += loss_total.item()
            bce_accum += loss_bce.item()
            concept_accum += loss_concept.item()
            mono_accum += loss_mono.item()

        scheduler.step()
        n_batches = len(train_loader)
        if epoch % 3 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | "
                f"Total: {total_loss_accum / n_batches:.4f} | "
                f"BCE: {bce_accum / n_batches:.4f} | "
                f"Concept: {concept_accum / n_batches:.4f} | "
                f"Mono: {mono_accum / n_batches:.4f}",
                flush=True,
            )

    # 5. Extract Logits and Post-Hoc Temperature Calibration
    print("\nExtracting logits for calibration and evaluation...", flush=True)
    val_logits, val_concepts, val_y = get_logits_and_labels(model, val_loader, DEVICE)
    test_logits, test_concepts, test_y = get_logits_and_labels(model, test_loader, DEVICE)

    scaler = TemperatureScaler()
    opt_temp = scaler.fit(val_logits, val_y)
    val_probs = scaler.predict_proba(val_logits)
    test_probs = scaler.predict_proba(test_logits)
    raw_test_probs = 1.0 / (1.0 + np.exp(-test_logits))

    raw_ece = expected_calibration_error(raw_test_probs, test_y)
    cal_ece = expected_calibration_error(test_probs, test_y)
    print(f"Optimal Calibration Temperature T*: {opt_temp:.3f}", flush=True)
    print(f"Held-Out Cohort Raw ECE:            {raw_ece:.4f} -> Calibrated ECE: {cal_ece:.4f}", flush=True)

    # 6. Split Conformal Abstention
    conformal = SplitConformalAbstainer(alpha=alpha)
    q_hat = conformal.calibrate(val_probs, val_y)
    conf_res = conformal.predict_with_abstention(test_probs)

    confident_mask = conf_res["confident_mask"]
    coverage = conf_res["coverage"]

    # Metrics on 100% test coverage
    full_metrics = compute_all_metrics(test_probs, test_y)
    # Metrics on confident subset (abstaining on ambiguous cases)
    if np.sum(confident_mask) > 10 and len(np.unique(test_y[confident_mask])) > 1:
        accepted_metrics = compute_all_metrics(test_probs[confident_mask], test_y[confident_mask])
    else:
        accepted_metrics = full_metrics.copy()

    # Bootstrap 95% Confidence Intervals
    print("\nComputing 95% Bootstrap Confidence Intervals (1,000 resamples)...", flush=True)
    boot_ci = bootstrap_confidence_intervals(test_probs, test_y, n_bootstraps=1000, seed=SEED)

    print("\n" + "=" * 80, flush=True)
    print("TABLE 3: PERFORMANCE ON ZERO-SHOT HELD-OUT COHORT", flush=True)
    print("=" * 80, flush=True)
    print(f"Coverage:                                100.0% (Full Cohort) vs {coverage * 100:.1f}% (Conformal Accepted)", flush=True)
    print(f"AUROC (100% Coverage):                   {full_metrics['auroc']:.4f} (95% CI: [{boot_ci['auroc'][1]:.4f}, {boot_ci['auroc'][2]:.4f}])", flush=True)
    print(f"AUROC on Conformal Accepted Patients:    {accepted_metrics['auroc']:.4f}", flush=True)
    print(f"Accuracy (100% Coverage):                {full_metrics['accuracy'] * 100:.2f}%", flush=True)
    print(f"Accuracy on Conformal Accepted:          {accepted_metrics['accuracy'] * 100:.2f}%", flush=True)
    print(f"Brier Score:                             {full_metrics['brier']:.4f} (95% CI: [{boot_ci['brier'][1]:.4f}, {boot_ci['brier'][2]:.4f}])", flush=True)
    print(f"Expected Calibration Error (ECE):        {full_metrics['ece']:.4f} (95% CI: [{boot_ci['ece'][1]:.4f}, {boot_ci['ece'][2]:.4f}])", flush=True)
    print(f"Sensitivity (Recall):                    {full_metrics['sensitivity']:.4f}", flush=True)
    print(f"Specificity:                             {full_metrics['specificity']:.4f}", flush=True)
    print("=" * 80, flush=True)

    # Save Table 3
    table3 = pd.DataFrame([
        {
            "Protocol": f"LOCO Zero-Shot ({held_out_cohort})",
            "Model": "SchemaFlexTabNet-QC (Full Coverage)",
            "Coverage (%)": 100.0,
            "AUROC": full_metrics["auroc"],
            "AUROC_95CI_Low": boot_ci["auroc"][1],
            "AUROC_95CI_High": boot_ci["auroc"][2],
            "Accuracy (%)": full_metrics["accuracy"] * 100,
            "Brier Score": full_metrics["brier"],
            "ECE": full_metrics["ece"],
            "Sensitivity": full_metrics["sensitivity"],
            "Specificity": full_metrics["specificity"],
        },
        {
            "Protocol": f"LOCO Zero-Shot ({held_out_cohort})",
            "Model": f"SchemaFlexTabNet-QC (Conformal alpha={alpha:.2f})",
            "Coverage (%)": coverage * 100,
            "AUROC": accepted_metrics["auroc"],
            "AUROC_95CI_Low": accepted_metrics["auroc"],
            "AUROC_95CI_High": accepted_metrics["auroc"],
            "Accuracy (%)": accepted_metrics["accuracy"] * 100,
            "Brier Score": accepted_metrics["brier"],
            "ECE": accepted_metrics["ece"],
            "Sensitivity": accepted_metrics["sensitivity"],
            "Specificity": accepted_metrics["specificity"],
        },
    ])
    table3_path = os.path.join(results_dir, "table3_proposed_loco_conformal.csv")
    table3.to_csv(table3_path, index=False)

    # 7. Risk-Coverage Curve (Sweep alpha in [0.02, 0.30])
    risk_coverage_records = []
    for a in np.linspace(0.02, 0.30, 15):
        c_inst = SplitConformalAbstainer(alpha=a)
        c_inst.calibrate(val_probs, val_y)
        res = c_inst.predict_with_abstention(test_probs)
        c_mask = res["confident_mask"]
        c_cov = res["coverage"]
        if np.sum(c_mask) > 10 and len(np.unique(test_y[c_mask])) > 1:
            c_auc = roc_auc_score(test_y[c_mask], test_probs[c_mask])
            c_acc = accuracy_score(test_y[c_mask], (test_probs[c_mask] >= 0.5).astype(int))
        else:
            c_auc = full_metrics["auroc"]
            c_acc = full_metrics["accuracy"]
        risk_coverage_records.append({
            "alpha": float(a),
            "coverage_pct": float(c_cov * 100),
            "auroc": float(c_auc),
            "accuracy_pct": float(c_acc * 100),
        })
    df_rc = pd.DataFrame(risk_coverage_records)
    rc_path = os.path.join(results_dir, "risk_coverage_curve.csv")
    df_rc.to_csv(rc_path, index=False)

    # 8. Clinical Subgroup Fairness Audit across Sex and Age Bands
    print("\nConducting Clinical Subgroup Fairness Audit...", flush=True)
    fairness_df = subgroup_fairness_audit(test_df, test_probs, target_col="target", threshold=0.5)
    fairness_path = os.path.join(results_dir, "fairness_subgroup_audit.csv")
    fairness_df.to_csv(fairness_path, index=False)
    print(fairness_df.to_string(index=False), flush=True)

    # 9. Decision Curve Analysis (DCA Net Benefit)
    print("\nComputing Decision Curve Analysis (Net Benefit)...", flush=True)
    dca_df = decision_curve_net_benefit(test_probs, test_y)
    dca_path = os.path.join(results_dir, "decision_curve_analysis.csv")
    dca_df.to_csv(dca_path, index=False)
    print(dca_df.head(6).to_string(index=False), flush=True)

    print(f"\nAll Phase 4 & 5 outputs saved to: {results_dir}", flush=True)
    return {
        "table3": table3,
        "risk_coverage": df_rc,
        "fairness": fairness_df,
        "dca": dca_df,
    }


if __name__ == "__main__":
    train_loco_pipeline()
