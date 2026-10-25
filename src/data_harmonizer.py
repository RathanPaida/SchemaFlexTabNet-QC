"""
Module 1: Physiological Quality Control & Cross-Cohort Data Harmonizer
Harmonizes heterogeneous EHR and clinical examination datasets into a canonical schema
without requiring lossy inner-joins.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Canonical feature lists
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

ALL_CANONICAL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES


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
        """
        Deduplicates rows and removes physiologically impossible values based on clinical bounds.
        """
        initial_rows = len(df)
        cleaned = df.copy()

        # 1. Exact duplicate removal
        feature_cols = [c for c in cleaned.columns if c not in ["patient_id", "cohort", "target"]]
        cleaned = cleaned.drop_duplicates(subset=feature_cols, keep="first")
        after_dedup = len(cleaned)

        # 2. Physiological range checks
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
        """Derives harmonized clinical meta-features and auxiliary concept labels."""
        df = df.copy()

        # Pulse pressure: SBP - DBP
        if "systolic_bp" in df.columns and "diastolic_bp" in df.columns:
            both_valid = df["systolic_bp"].notna() & df["diastolic_bp"].notna()
            df.loc[both_valid, "pulse_pressure"] = (
                df.loc[both_valid, "systolic_bp"] - df.loc[both_valid, "diastolic_bp"]
            )

        # Harmonize continuous cholesterol (mg/dL) into ATP-III ordinal tiers (1, 2, 3)
        if "chol_mgdl" in df.columns:
            mask_has_chol = df["chol_mgdl"].notna()
            # If chol_tier not already set, map from continuous
            if "chol_tier" not in df.columns:
                df["chol_tier"] = np.nan
            df.loc[mask_has_chol & df["chol_tier"].isna(), "chol_tier"] = pd.cut(
                df.loc[mask_has_chol & df["chol_tier"].isna(), "chol_mgdl"],
                bins=[-np.inf, 200.0, 239.0, np.inf],
                labels=[1, 2, 3],
            ).astype(float)

        # Auxiliary Clinical Concept: Stage 2 Hypertension (SBP >= 140 or DBP >= 90)
        sbp_high = df["systolic_bp"] >= 140.0 if "systolic_bp" in df.columns else pd.Series(False, index=df.index)
        dbp_high = df["diastolic_bp"] >= 90.0 if "diastolic_bp" in df.columns else pd.Series(False, index=df.index)
        df["concept_hypertension"] = (sbp_high | dbp_high).astype(float)

        return df


def load_raw_kaggle_heart(data_dir: str) -> pd.DataFrame:
    """Loads Kaggle Heart Disease dataset (Johnsmith88 copy)."""
    path = os.path.join(data_dir, "heart_kaggle_johnsmith88.csv")
    df = pd.read_csv(path)
    mapped = pd.DataFrame()
    mapped["cohort"] = ["Kaggle_Heart_Johnsmith88"] * len(df)
    mapped["age_years"] = df["age"].astype(float)
    mapped["sex_male"] = df["sex"].astype(float)
    mapped["chest_pain_type"] = (df["cp"] + 1).astype(float)  # 1-4 scale
    mapped["systolic_bp"] = df["trestbps"].astype(float)
    mapped["chol_mgdl"] = df["chol"].astype(float)
    mapped["max_hr_thalach"] = df["thalach"].astype(float)
    mapped["exercise_angina"] = df["exang"].astype(float)
    mapped["st_depression_oldpeak"] = df["oldpeak"].astype(float)
    mapped["target"] = df["target"].astype(float)
    return mapped


def load_raw_sulianova(data_dir: str) -> pd.DataFrame:
    """Loads Sulianova Cardiovascular Disease dataset (70,000 rows)."""
    path = os.path.join(data_dir, "cardio_sulianova.csv")
    df = pd.read_csv(path, sep=";")
    mapped = pd.DataFrame()
    mapped["cohort"] = ["Sulianova_CVD"] * len(df)
    mapped["age_years"] = (df["age"] / 365.25).astype(float)
    mapped["sex_male"] = (df["gender"] == 2).astype(float)  # 1: female, 2: male
    mapped["systolic_bp"] = df["ap_hi"].astype(float)
    mapped["diastolic_bp"] = df["ap_lo"].astype(float)
    
    # BMI from height (cm) and weight (kg)
    height_m = df["height"] / 100.0
    mapped["bmi"] = (df["weight"] / (height_m ** 2)).astype(float)
    
    mapped["chol_tier"] = df["cholesterol"].astype(float)  # 1, 2, 3
    mapped["gluc_tier"] = df["gluc"].astype(float)         # 1, 2, 3
    mapped["smoker"] = df["smoke"].astype(float)
    mapped["target"] = df["cardio"].astype(float)
    return mapped


def load_raw_framingham(data_dir: str) -> pd.DataFrame:
    """Loads Framingham Heart Study teaching subset (4,240 rows)."""
    path = os.path.join(data_dir, "framingham.csv")
    df = pd.read_csv(path)
    mapped = pd.DataFrame()
    mapped["cohort"] = ["Framingham_Heart_Study"] * len(df)
    mapped["age_years"] = df["age"].astype(float)
    mapped["sex_male"] = df["male"].astype(float)
    mapped["smoker"] = df["currentSmoker"].astype(float)
    mapped["chol_mgdl"] = df["totChol"].astype(float)
    mapped["systolic_bp"] = df["sysBP"].astype(float)
    mapped["diastolic_bp"] = df["diaBP"].astype(float)
    mapped["bmi"] = df["BMI"].astype(float)
    
    # Glucose tier: <100 normal (1), 100-125 prediabetes (2), >=126 diabetes (3)
    if "glucose" in df.columns:
        g_tier = pd.cut(df["glucose"], bins=[-np.inf, 99.0, 125.0, np.inf], labels=[1, 2, 3]).astype(float)
        mapped["gluc_tier"] = g_tier
    
    mapped["target"] = df["TenYearCHD"].astype(float)
    return mapped


def load_raw_uci_heart(data_dir: str) -> pd.DataFrame:
    """Loads all 4 UCI Heart Disease cohorts (Cleveland, Hungarian, Switzerland, VA Long Beach)."""
    cohorts_meta = [
        ("cleveland", "UCI_Cleveland"),
        ("hungarian", "UCI_Hungarian"),
        ("switzerland", "UCI_Switzerland"),
        ("va", "UCI_VA_Long_Beach"),
    ]
    cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    all_dfs = []
    for fname, cname in cohorts_meta:
        path = os.path.join(data_dir, f"heart_uci_{fname}.data")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, header=None, names=cols, na_values=["?"])
        # In Hungarian, Switzerland, and VA, 0 indicates missing trestbps or chol
        df["trestbps"] = df["trestbps"].replace(0, np.nan)
        df["chol"] = df["chol"].replace(0, np.nan)
        df["thalach"] = df["thalach"].replace(0, np.nan)

        mapped = pd.DataFrame()
        mapped["cohort"] = [cname] * len(df)
        mapped["age_years"] = pd.to_numeric(df["age"], errors="coerce")
        mapped["sex_male"] = pd.to_numeric(df["sex"], errors="coerce")
        mapped["chest_pain_type"] = pd.to_numeric(df["cp"], errors="coerce")
        mapped["systolic_bp"] = pd.to_numeric(df["trestbps"], errors="coerce")
        mapped["chol_mgdl"] = pd.to_numeric(df["chol"], errors="coerce")
        mapped["max_hr_thalach"] = pd.to_numeric(df["thalach"], errors="coerce")
        mapped["exercise_angina"] = pd.to_numeric(df["exang"], errors="coerce")
        mapped["st_depression_oldpeak"] = pd.to_numeric(df["oldpeak"], errors="coerce")
        # In UCI, num: 0 = healthy, >0 = coronary disease
        raw_num = pd.to_numeric(df["num"], errors="coerce")
        mapped["target"] = (raw_num > 0).astype(float)
        all_dfs.append(mapped)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def load_raw_statlog(data_dir: str) -> pd.DataFrame:
    """Loads UCI Statlog (Heart) dataset."""
    path = os.path.join(data_dir, "statlog_heart.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    
    # Map possible column naming variations
    col_map = {
        "rest-bp": "systolic_bp",
        "resting_blood_pressure": "systolic_bp",
        "serum-chol": "chol_mgdl",
        "serum_cholestoral": "chol_mgdl",
        "max-heart-rate": "max_hr_thalach",
        "maximum_heart_rate_achieved": "max_hr_thalach",
        "chest-pain": "chest_pain_type",
        "chest_pain_type": "chest_pain_type",
        "angina": "exercise_angina",
        "exercise_induced_angina": "exercise_angina",
    }
    df = df.rename(columns=col_map)
    mapped = pd.DataFrame()
    mapped["cohort"] = ["UCI_Statlog_Heart"] * len(df)
    mapped["age_years"] = df["age"].astype(float)
    mapped["sex_male"] = df["sex"].astype(float)
    mapped["chest_pain_type"] = df.get("chest_pain_type", pd.Series(np.nan, index=df.index)).astype(float)
    mapped["systolic_bp"] = df.get("systolic_bp", pd.Series(np.nan, index=df.index)).astype(float)
    mapped["chol_mgdl"] = df.get("chol_mgdl", pd.Series(np.nan, index=df.index)).astype(float)
    mapped["max_hr_thalach"] = df.get("max_hr_thalach", pd.Series(np.nan, index=df.index)).astype(float)
    mapped["exercise_angina"] = df.get("exercise_angina", pd.Series(np.nan, index=df.index)).astype(float)
    mapped["st_depression_oldpeak"] = df.get("oldpeak", pd.Series(np.nan, index=df.index)).astype(float)
    # Statlog target: 1 = absence (0), 2 = presence (1)
    mapped["target"] = (df["target"] == 2).astype(float)
    return mapped


def load_raw_z_sani(data_dir: str) -> pd.DataFrame:
    """Loads Z-Alizadeh Sani coronary artery disease dataset."""
    path = os.path.join(data_dir, "z_alizadeh_sani.xlsx")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path)
    mapped = pd.DataFrame()
    mapped["cohort"] = ["Z_Alizadeh_Sani"] * len(df)
    mapped["age_years"] = pd.to_numeric(df["Age"], errors="coerce")
    mapped["sex_male"] = (df["Sex"].astype(str).str.lower().str.startswith("m")).astype(float)
    mapped["systolic_bp"] = pd.to_numeric(df["BP"], errors="coerce")
    mapped["bmi"] = pd.to_numeric(df["BMI"], errors="coerce")
    mapped["smoker"] = pd.to_numeric(df["Current Smoker"], errors="coerce")
    # Target: Cath == 'Cad' -> 1, 'Normal' -> 0
    mapped["target"] = (df["Cath"].astype(str).str.strip().str.lower() == "cad").astype(float)
    return mapped


def harmonize_all_datasets(raw_dir: str, processed_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """
    Executes Module 1 harmonization across all raw datasets:
    1. Standardizes schemas to canonical variables
    2. Applies clinical plausibility filtering and deduplication
    3. Derives auxiliary clinical concepts (pulse pressure, chol tiers, hypertension)
    4. Saves clean multi-cohort datasets
    """
    os.makedirs(processed_dir, exist_ok=True)
    raw_loaders = [
        ("UCI Heart 4-Cohorts", lambda: load_raw_uci_heart(raw_dir)),
        ("Sulianova Cardiovascular", lambda: load_raw_sulianova(raw_dir)),
        ("Framingham Heart Study", lambda: load_raw_framingham(raw_dir)),
        ("UCI Statlog Heart", lambda: load_raw_statlog(raw_dir)),
        ("Z-Alizadeh Sani", lambda: load_raw_z_sani(raw_dir)),
    ]

    cleaned_dfs = []
    audit_logs = []

    for label, loader in raw_loaders:
        raw_df = loader()
        if raw_df.empty:
            continue
        # Split by individual cohort in case of multi-cohort loader
        for cname, group in raw_df.groupby("cohort"):
            clean_grp, audit = ClinicalPlausibilityGatekeeper.audit_and_clean(group, cname)
            clean_grp = ClinicalPlausibilityGatekeeper.engineer_clinical_concepts(clean_grp)
            cleaned_dfs.append(clean_grp)
            audit_logs.append(audit)

    clean_multi_cohort = pd.concat(cleaned_dfs, ignore_index=True)
    
    # Save processed clean datasets
    parquet_path = os.path.join(processed_dir, "clean_multi_cohort.parquet")
    csv_path = os.path.join(processed_dir, "clean_multi_cohort.csv")
    clean_multi_cohort.to_parquet(parquet_path, index=False)
    clean_multi_cohort.to_csv(csv_path, index=False)

    # Also load and clean Kaggle Heart duplicate dataset separately for forensic audit
    raw_kaggle = load_raw_kaggle_heart(raw_dir)
    clean_kaggle, audit_kaggle = ClinicalPlausibilityGatekeeper.audit_and_clean(raw_kaggle, "Kaggle_Heart_Audit")
    clean_kaggle = ClinicalPlausibilityGatekeeper.engineer_clinical_concepts(clean_kaggle)
    audit_logs.append(audit_kaggle)

    # Save audit logs
    audit_df = pd.DataFrame(audit_logs)
    audit_df.to_csv(os.path.join(processed_dir, "harmonization_audit_log.csv"), index=False)

    return clean_multi_cohort, clean_kaggle, audit_logs


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r_dir = os.path.join(base_dir, "data", "raw")
    p_dir = os.path.join(base_dir, "data", "processed")
    clean_df, _, logs = harmonize_all_datasets(r_dir, p_dir)
    print("Harmonization completed successfully!")
    print(pd.DataFrame(logs).to_string(index=False))
