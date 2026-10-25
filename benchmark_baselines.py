"""
Phase 3: Fair Baseline Benchmarking
Trains and evaluates 7 classical tabular and deep learning architectures:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM
5. CatBoost
6. TabNet
7. Multi-Layer Perceptron (MLP)

Evaluated under:
- Two Feature Sets: (a) Raw Features Only, (b) Raw + Engineered Meta-Features
- Two Evaluation Protocols: (1) 5-Fold Stratified Cross-Validation, (2) Leave-One-Cohort-Out (LOCO)
Produces: results/table2_baseline_benchmark.csv
"""

import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from src.evaluate_metrics import compute_all_metrics

SEED = 42
np.random.seed(SEED)

RAW_FEATURES = [
    "age_years",
    "sex_male",
    "systolic_bp",
    "diastolic_bp",
    "chol_mgdl",
    "max_hr_thalach",
    "st_depression_oldpeak",
    "chest_pain_type",
    "exercise_angina",
    "smoker",
]

ENGINEERED_META_FEATURES = [
    "bmi",
    "pulse_pressure",
    "chol_tier",
    "gluc_tier",
    "concept_hypertension",
]

ALL_BENCHMARK_FEATURES = RAW_FEATURES + ENGINEERED_META_FEATURES


def get_models(seed: int = SEED) -> Dict[str, object]:
    """Returns instances of all 7 baseline models with standardized configurations."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=seed, class_weight="balanced", n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=seed, eval_metric="logloss", n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=seed, verbose=-1, n_jobs=-1),
        "CatBoost": CatBoostClassifier(iterations=150, depth=5, learning_rate=0.05, random_seed=seed, verbose=0, thread_count=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=seed, early_stopping=True),
        "TabNet": TabNetClassifier(n_d=16, n_a=16, n_steps=3, gamma=1.3, seed=seed, verbose=0),
    }


def preprocess_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Imputes missing values and scales features for linear/neural models."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    X_train_scl = scaler.fit_transform(X_train_imp)
    X_test_scl = scaler.transform(X_test_imp)

    return X_train_scl, X_test_scl


def evaluate_model(
    model_name: str,
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Fits model and computes AUROC, AUPRC, Accuracy, F1, Brier, and ECE."""
    start = time.time()
    if model_name == "TabNet":
        y_train_int = y_train.astype(int)
        model.fit(
            X_train, y_train_int,
            eval_set=[(X_test, y_test.astype(int))],
            patience=5, max_epochs=15, batch_size=512,
        )
        probs = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

    elapsed = time.time() - start
    metrics = compute_all_metrics(probs, y_test)
    metrics["train_time_sec"] = elapsed
    return metrics


def run_baseline_benchmarks() -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    df_path = os.path.join(processed_dir, "clean_multi_cohort.parquet")
    df = pd.read_parquet(df_path)

    print("=" * 80)
    print("PHASE 3: FAIR BASELINE BENCHMARKING ACROSS 7 ARCHITECTURES")
    print(f"Total benchmark samples: {len(df):,} across {df['cohort'].nunique()} cohorts.")
    print("=" * 80)

    # For faster, responsive benchmarking while preserving statistical rigor,
    # sample Sulianova to 15,000 if running full cross-validation
    sampled_dfs = []
    for cname, grp in df.groupby("cohort"):
        if cname == "Sulianova_CVD" and len(grp) > 15000:
            sampled_dfs.append(grp.sample(n=15000, random_state=SEED))
        else:
            sampled_dfs.append(grp)
    bench_df = pd.concat(sampled_dfs, ignore_index=True)

    configs = [
        ("Raw Features", [c for c in RAW_FEATURES if c in bench_df.columns]),
        ("Raw + Meta-Features", [c for c in ALL_BENCHMARK_FEATURES if c in bench_df.columns]),
    ]

    all_results = []

    # PROTOCOL 1: Stratified 5-Fold Cross Validation
    print("\n--- Protocol 1: Stratified 5-Fold Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for cfg_name, feat_cols in configs:
        print(f"\nEvaluating Configuration: {cfg_name} ({len(feat_cols)} features)...")
        for model_name in get_models(SEED).keys():
            fold_metrics = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(bench_df, bench_df["target"]), 1):
                train_fold = bench_df.iloc[train_idx]
                val_fold = bench_df.iloc[val_idx]

                X_tr, X_va = preprocess_data(train_fold[feat_cols], val_fold[feat_cols])
                y_tr, y_va = train_fold["target"].values, val_fold["target"].values

                mdl = get_models(SEED)[model_name]
                m = evaluate_model(model_name, mdl, X_tr, y_tr, X_va, y_va)
                fold_metrics.append(m)

            avg_auroc = np.mean([m["auroc"] for m in fold_metrics])
            avg_auprc = np.mean([m["auprc"] for m in fold_metrics])
            avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
            avg_f1 = np.mean([m["f1"] for m in fold_metrics])
            avg_brier = np.mean([m["brier"] for m in fold_metrics])
            avg_ece = np.mean([m["ece"] for m in fold_metrics])

            print(f"  [{model_name:<19}] AUROC: {avg_auroc:.4f} | Acc: {avg_acc*100:.2f}% | Brier: {avg_brier:.4f} | ECE: {avg_ece:.4f}")
            all_results.append({
                "Protocol": "5-Fold Stratified CV",
                "Configuration": cfg_name,
                "Model Architecture": model_name,
                "AUROC": avg_auroc,
                "AUPRC": avg_auprc,
                "Accuracy (%)": avg_acc * 100,
                "F1-Score": avg_f1,
                "Brier Score": avg_brier,
                "ECE": avg_ece,
            })

    # PROTOCOL 2: Leave-One-Cohort-Out (LOCO) Transfer
    # Train on UCI_Cleveland + UCI_Hungarian + Sulianova + Framingham; test on held-out cohorts
    print("\n--- Protocol 2: Leave-One-Cohort-Out (LOCO) Zero-Shot Transfer ---")
    held_out_cohorts = ["UCI_Statlog_Heart", "Z_Alizadeh_Sani", "UCI_VA_Long_Beach"]

    for held_out in held_out_cohorts:
        train_df = bench_df[bench_df["cohort"] != held_out]
        test_df = bench_df[bench_df["cohort"] == held_out]
        if test_df.empty or len(test_df["target"].unique()) < 2:
            continue

        print(f"\nHeld-Out Test Cohort: {held_out} (N={len(test_df)})")
        feat_cols = [c for c in ALL_BENCHMARK_FEATURES if c in bench_df.columns]
        X_tr, X_te = preprocess_data(train_df[feat_cols], test_df[feat_cols])
        y_tr, y_te = train_df["target"].values, test_df["target"].values

        for model_name, mdl in get_models(SEED).items():
            m = evaluate_model(model_name, mdl, X_tr, y_tr, X_te, y_te)
            print(f"  [{model_name:<19}] Zero-Shot AUROC: {m['auroc']:.4f} | Acc: {m['accuracy']*100:.2f}% | Brier: {m['brier']:.4f}")
            all_results.append({
                "Protocol": f"LOCO Zero-Shot ({held_out})",
                "Configuration": "Raw + Meta-Features",
                "Model Architecture": model_name,
                "AUROC": m["auroc"],
                "AUPRC": m["auprc"],
                "Accuracy (%)": m["accuracy"] * 100,
                "F1-Score": m["f1"],
                "Brier Score": m["brier"],
                "ECE": m["ece"],
            })

    results_df = pd.DataFrame(all_results)
    out_csv = os.path.join(results_dir, "table2_baseline_benchmark.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved Table 2 Baseline Benchmark to: {out_csv}")
    print("=" * 80)
    return results_df


if __name__ == "__main__":
    run_baseline_benchmarks()
