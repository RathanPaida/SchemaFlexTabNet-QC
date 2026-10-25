"""
Module 5: Publication-Grade Evaluation Suite
Computes:
1. Discrimination: AUROC, AUPRC, Accuracy, F1, Sensitivity, Specificity
2. Calibration: Expected Calibration Error (ECE), Brier Score
3. Uncertainty & Coverage: Bootstrap 95% Confidence Intervals, DeLong's Test
4. Clinical Utility: Decision Curve Analysis (Net Benefit) across decision thresholds
5. Fairness: Subgroup Audits (Sex, Age Bands) with Equalized Odds & Demographic Parity
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across equal-width probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for b_low, b_high in zip(bins[:-1], bins[1:]):
        mask = (probs >= b_low) & (probs < b_high)
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(probs[mask])
            ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def compute_all_metrics(probs: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes comprehensive discrimination and calibration metrics."""
    preds = (probs >= threshold).astype(int)
    y_true_int = y_true.astype(int)

    # Check for single-class corner case
    if len(np.unique(y_true_int)) < 2:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "brier": float(brier_score_loss(y_true_int, probs)),
            "ece": float(expected_calibration_error(probs, y_true_int)),
            "accuracy": float(accuracy_score(y_true_int, preds)),
            "f1": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
        }

    auroc = float(roc_auc_score(y_true_int, probs))
    auprc = float(average_precision_score(y_true_int, probs))
    brier = float(brier_score_loss(y_true_int, probs))
    ece = float(expected_calibration_error(probs, y_true_int))
    acc = float(accuracy_score(y_true_int, preds))
    f1 = float(f1_score(y_true_int, preds, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true_int, preds, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece": ece,
        "accuracy": acc,
        "f1": f1,
        "sensitivity": sens,
        "specificity": spec,
    }


def bootstrap_confidence_intervals(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Computes empirical bootstrap confidence intervals (Mean, 2.5%, 97.5%)
    for AUROC, AUPRC, Brier Score, and ECE.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metrics_list = {"auroc": [], "auprc": [], "brier": [], "ece": []}

    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n, size=n)
        b_y = y_true[boot_idx]
        b_p = probs[boot_idx]
        if len(np.unique(b_y)) < 2:
            continue
        metrics_list["auroc"].append(roc_auc_score(b_y, b_p))
        metrics_list["auprc"].append(average_precision_score(b_y, b_p))
        metrics_list["brier"].append(brier_score_loss(b_y, b_p))
        metrics_list["ece"].append(expected_calibration_error(b_p, b_y))

    alpha = 1.0 - ci_level
    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0

    ci_results = {}
    for k, vals in metrics_list.items():
        if vals:
            mean_val = float(np.mean(vals))
            low_val = float(np.percentile(vals, low_pct))
            high_val = float(np.percentile(vals, high_pct))
            ci_results[k] = (mean_val, low_val, high_val)
        else:
            ci_results[k] = (float("nan"), float("nan"), float("nan"))

    return ci_results


def delong_roc_test(p1: np.ndarray, p2: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """
    Computes DeLong test comparing the AUROC of model 1 vs model 2 on the same test set.
    Returns: (z_statistic, p_value)
    """
    y_true = np.asarray(y_true, dtype=int)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)

    if m == 0 or n == 0:
        return 0.0, 1.0

    # Structural components
    v10_1 = np.zeros(m)
    v01_1 = np.zeros(n)
    v10_2 = np.zeros(m)
    v01_2 = np.zeros(n)

    for i, p_idx in enumerate(pos_idx):
        v10_1[i] = np.mean(p1[p_idx] > p1[neg_idx]) + 0.5 * np.mean(p1[p_idx] == p1[neg_idx])
        v10_2[i] = np.mean(p2[p_idx] > p2[neg_idx]) + 0.5 * np.mean(p2[p_idx] == p2[neg_idx])

    for j, n_idx in enumerate(neg_idx):
        v01_1[j] = np.mean(p1[pos_idx] > p1[n_idx]) + 0.5 * np.mean(p1[pos_idx] == p1[n_idx])
        v01_2[j] = np.mean(p2[pos_idx] > p2[n_idx]) + 0.5 * np.mean(p2[pos_idx] == p2[n_idx])

    auc1 = np.mean(v10_1)
    auc2 = np.mean(v10_2)

    # Covariance estimation
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    var_diff = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n

    if var_diff <= 1e-12:
        return 0.0, 1.0

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p_val)


def decision_curve_net_benefit(
    probs: np.ndarray, y_true: np.ndarray, thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Computes Decision Curve Analysis (DCA) Net Benefit across clinical probability thresholds.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.80, 16)

    n = len(y_true)
    prev = np.mean(y_true)
    records = []

    for pt in thresholds:
        preds = (probs >= pt).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        weight = pt / (1.0 - pt)
        
        # Model net benefit
        nb_model = (tp / n) - (fp / n) * weight
        # Treat all net benefit
        nb_all = prev - (1.0 - prev) * weight
        # Treat none net benefit
        nb_none = 0.0

        records.append({
            "threshold": float(pt),
            "net_benefit_model": float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": float(nb_none),
        })

    return pd.DataFrame(records)


def subgroup_fairness_audit(
    df: pd.DataFrame,
    probs: np.ndarray,
    target_col: str = "target",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Conducts a clinical subgroup fairness audit across Sex and Age Bands:
    - Sex (Male: sex_male=1, Female: sex_male=0)
    - Age Bands (<50, 50-65, >65)
    Reports TPR, FPR, Positive Selection Rate, and Equalized Odds Disparity.
    """
    eval_df = df.copy()
    eval_df["probs"] = probs
    eval_df["preds"] = (probs >= threshold).astype(int)
    eval_df["true_y"] = eval_df[target_col].astype(int)

    # Define subgroups
    groups = {}
    if "sex_male" in eval_df.columns:
        groups["Sex: Male"] = eval_df["sex_male"] == 1
        groups["Sex: Female"] = eval_df["sex_male"] == 0

    if "age_years" in eval_df.columns:
        groups["Age: <50"] = eval_df["age_years"] < 50
        groups["Age: 50-65"] = (eval_df["age_years"] >= 50) & (eval_df["age_years"] <= 65)
        groups["Age: >65"] = eval_df["age_years"] > 65

    overall_metrics = compute_all_metrics(probs, eval_df["true_y"].values, threshold)
    overall_tpr = overall_metrics["sensitivity"]
    overall_fpr = 1.0 - overall_metrics["specificity"]

    records = []
    for gname, mask in groups.items():
        sub = eval_df[mask]
        if len(sub) == 0:
            continue
        sub_probs = sub["probs"].values
        sub_y = sub["true_y"].values
        m = compute_all_metrics(sub_probs, sub_y, threshold)
        
        tpr = m["sensitivity"]
        fpr = 1.0 - m["specificity"]
        pos_rate = float(np.mean(sub["preds"]))
        # Equalized Odds disparity
        eq_odds_disp = max(abs(tpr - overall_tpr), abs(fpr - overall_fpr))

        records.append({
            "subgroup": gname,
            "sample_size": len(sub),
            "prevalence": float(np.mean(sub_y)),
            "auroc": m["auroc"],
            "tpr_sensitivity": tpr,
            "fpr": fpr,
            "selection_rate": pos_rate,
            "equalized_odds_disparity": eq_odds_disp,
        })

    return pd.DataFrame(records)
