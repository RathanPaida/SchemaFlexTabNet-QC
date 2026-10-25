"""
Master End-to-End Execution Pipeline
Runs all 5 phases of the research blueprint:
1. Automated Dataset Download & Integrity Verification (data/raw/)
2. Multi-Cohort Harmonization & Physiological Gatekeeper (src/data_harmonizer.py)
3. Phase 1 Forensic Audit of Baseline Flaws (baseline_audit.py -> results/table1_forensic_audit.csv)
4. Phase 3 Fair Baseline Benchmarking across 7 Models (benchmark_baselines.py -> results/table2_baseline_benchmark.csv)
5. Phase 4 & 5 Proposed SchemaFlexTabNet-QC Pipeline (train_loco_pipeline.py -> results/table3_proposed_loco_conformal.csv)
"""

import os
import sys
import time
import pandas as pd

from download_datasets import download_all_datasets
from src.data_harmonizer import harmonize_all_datasets
from baseline_audit import run_forensic_audit
from benchmark_baselines import run_baseline_benchmarks
from train_loco_pipeline import train_loco_pipeline


def run_complete_research_pipeline():
    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results")

    print("=" * 80)
    print("STARTING COMPLETE END-TO-END CROSS-COHORT RESEARCH PIPELINE")
    print("=" * 80)

    # Step 1: Download Datasets
    print("\n>>> STEP 1: DOWNLOADING DATASETS...")
    download_res = download_all_datasets()

    # Step 2: Data Harmonization & Physiological Quality Control
    print("\n>>> STEP 2: HARMONIZING COHORTS & PHYSIOLOGICAL QUALITY CONTROL...")
    clean_df, audit_df, logs = harmonize_all_datasets(raw_dir, processed_dir)
    print(f"Harmonized dataset shape: {clean_df.shape} across {clean_df['cohort'].nunique()} cohorts.")

    # Step 3: Phase 1 Forensic Audit
    print("\n>>> STEP 3: EXECUTING PHASE 1 FORENSIC AUDIT...")
    table1 = run_forensic_audit()

    # Step 4: Phase 3 Fair Baseline Benchmarks
    print("\n>>> STEP 4: EXECUTING PHASE 3 FAIR BASELINE BENCHMARKS...")
    table2 = run_baseline_benchmarks()

    # Step 5: Phase 4 & 5 Proposed Model Training & Evaluation
    print("\n>>> STEP 5: EXECUTING PHASE 4 & 5 SCHEMAFLEXTABNET-QC TRAINING & CONFORMAL QC...")
    loco_results = train_loco_pipeline(held_out_cohort="UCI_Statlog_Heart", epochs=15, alpha=0.10)

    # Step 6: Generate Publication Figures
    print("\n>>> STEP 6: GENERATING PUBLICATION FIGURES (300 DPI)...")
    from generate_figures import plot_forensic_audit, plot_risk_coverage, plot_decision_curve, plot_fairness_subgroup
    plot_forensic_audit()
    plot_risk_coverage()
    plot_decision_curve()
    plot_fairness_subgroup()

    # Step 7: Multi-Cohort LOCO Matrix, 5-Stage Ablation & Missingness Stress Test
    print("\n>>> STEP 7: EXECUTING MULTI-COHORT LOCO MATRIX, ABLATION & MISSINGNESS STRESS TEST...")
    from run_ablation_and_loco_matrix import (
        load_and_audit_cross_cohort_duplicates,
        run_multi_cohort_loco_matrix,
        run_ablation_and_monotonicity_study,
        run_missingness_stress_test,
    )
    clean_dedup_df = load_and_audit_cross_cohort_duplicates()
    run_multi_cohort_loco_matrix(clean_dedup_df)
    run_ablation_and_monotonicity_study(clean_dedup_df)
    run_missingness_stress_test(clean_dedup_df)

    # Step 8: DeLong Significance, Mondrian Fairness & Federated Learning Simulation
    print("\n>>> STEP 8: EXECUTING DELONG SIGNIFICANCE, MONDRIAN FAIRNESS & FEDERATED LEARNING...")
    from run_federated_and_fairness_suite import (
        run_delong_statistical_testing,
        run_mondrian_group_fair_conformal,
        run_simulated_federated_learning,
    )
    run_delong_statistical_testing(clean_dedup_df)
    run_mondrian_group_fair_conformal(clean_dedup_df)
    run_simulated_federated_learning(clean_dedup_df)

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("RESEARCH PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Execution Time: {total_elapsed / 60:.2f} minutes")
    print(f"Artifacts and tables saved in: {results_dir}")
    print("\nGenerated Artifacts:")
    for f in sorted(os.listdir(results_dir)):
        p = os.path.join(results_dir, f)
        print(f"  - {f:<40} ({os.path.getsize(p):,} bytes)")
    print("=" * 80)


if __name__ == "__main__":
    run_complete_research_pipeline()
