"""
Generates Publication-Grade Figures (300 DPI) for the Research Paper:
1. figure1_forensic_audit.png: Accuracy and AUROC drop under true de-duplication
2. figure2_risk_coverage.png: Risk-Coverage Conformal Abstention Curve
3. figure3_decision_curve.png: Decision Curve Analysis (Clinical Net Benefit)
4. figure4_fairness_disparity.png: Subgroup Fairness Audit across Sex and Age Bands
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Use clean publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def plot_forensic_audit():
    csv_path = os.path.join(RESULTS_DIR, "table1_forensic_audit.csv")
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Leaked Split\n(Kaggle Copy)", "De-duplicated Split\n(True Patients)", "Sulianova Merge\n(Inner-Join Ceiling)"]
    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, df["Accuracy (%)"], width, label="Accuracy (%)", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, df["AUROC"] * 100, width, label="AUROC (x100)", color="#e07a5f")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Forensic Audit: Elimination of Artificial Accuracy from Duplicate Contamination", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(50, 105)
    ax.legend(loc="upper right")

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "figure1_forensic_audit.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_risk_coverage():
    csv_path = os.path.join(RESULTS_DIR, "risk_coverage_curve.csv")
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["coverage_pct"], df["auroc"], marker="o", color="#1d3557", linewidth=2.5, label="Conformal Zero-Shot AUROC")
    ax.axhline(0.8821, color="#e63946", linestyle="--", label="100% Coverage AUROC Baseline (0.8821)")

    ax.set_xlabel("Clinical Patient Coverage (%)", fontweight="bold")
    ax.set_ylabel("AUROC on Accepted Patients", fontweight="bold")
    ax.set_title("Risk-Coverage Abstention Curve: Safety Gatekeeper Under Conformal QC", fontweight="bold")
    ax.set_ylim(0.85, 0.95)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower left")

    # Annotate peak accepted AUROC
    ax.annotate(
        "Peak Accepted AUROC: 0.924\n(Ambiguous cases referred to clinician)",
        xy=(65.6, 0.924),
        xytext=(45, 0.932),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
        fontweight="bold",
        fontsize=9,
    )

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "figure2_risk_coverage.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_decision_curve():
    csv_path = os.path.join(RESULTS_DIR, "decision_curve_analysis.csv")
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["net_benefit_model"], color="#2a9d8f", linewidth=2.5, label="SchemaFlexTabNet-QC")
    ax.plot(df["threshold"], df["net_benefit_treat_all"], color="#e76f51", linestyle="--", linewidth=1.8, label="Treat All")
    ax.plot(df["threshold"], df["net_benefit_treat_none"], color="#264653", linestyle=":", linewidth=1.8, label="Treat None")

    ax.set_xlabel("Threshold Probability $p_t$ (Physician Risk Tolerance)", fontweight="bold")
    ax.set_ylabel("Net Benefit", fontweight="bold")
    ax.set_title("Decision Curve Analysis (DCA): Clinical Net Benefit", fontweight="bold")
    ax.set_ylim(-0.05, 0.50)
    ax.set_xlim(0.05, 0.70)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "figure3_decision_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fairness_subgroup():
    csv_path = os.path.join(RESULTS_DIR, "fairness_subgroup_audit.csv")
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df["auroc"], width, label="AUROC", color="#457b9d")
    ax.bar(x + width/2, df["tpr_sensitivity"], width, label="TPR (Sensitivity)", color="#a8dadc")

    ax.set_ylabel("Score")
    ax.set_title("Subgroup Fairness Audit across Demographic Partitions", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["subgroup"], rotation=15, ha="right")
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "figure4_fairness_disparity.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_forensic_audit()
    plot_risk_coverage()
    plot_decision_curve()
    plot_fairness_subgroup()
