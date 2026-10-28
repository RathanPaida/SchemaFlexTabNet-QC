#!/usr/bin/env python3
"""
build_pdf_manuscript.py
================================================================================
Compiles the comprehensive SchemaFlexTabNet-QC research manuscript into an
8-page IEEE JBHI / TBME-styled publication PDF with embedded high-resolution
figures, complete tables, verified equations, and all 22 academic references.
================================================================================
"""

import base64
import os
import subprocess
from pathlib import Path
import pypdf

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
HTML_OUT = BASE_DIR / "manuscript_print.html"
PDF_OUT = BASE_DIR / "SchemaFlexTabNet_QC_Research_Paper.pdf"

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(CHROME_PATH):
    CHROME_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def img_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


fig1_b64 = img_to_base64(RESULTS_DIR / "figure1_forensic_audit.png")
fig2_b64 = img_to_base64(RESULTS_DIR / "figure2_risk_coverage.png")
fig3_b64 = img_to_base64(RESULTS_DIR / "figure3_decision_curve.png")
fig4_b64 = img_to_base64(RESULTS_DIR / "figure4_fairness_disparity.png")
fig5_b64 = img_to_base64(RESULTS_DIR / "figure5_missingness_degradation.png")
fig6_b64 = img_to_base64(RESULTS_DIR / "figure6_mondrian_fairness_parity.png")
fig7_b64 = img_to_base64(RESULTS_DIR / "figure7_federated_vs_centralized.png")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Forensic Benchmarking and Schema-Flexible Clinical Tabular Learning</title>
<style>
  @page {{
    size: letter;
    margin: 14mm 12mm 14mm 12mm;
    @bottom-right {{
      content: counter(page);
    }}
  }}
  body {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 9.05pt;
    line-height: 1.28;
    color: #111;
    margin: 0;
    padding: 0;
    text-align: justify;
    text-justify: inter-word;
  }}
  .header {{
    text-align: center;
    margin-bottom: 11px;
  }}
  .journal-header {{
    font-size: 8pt;
    font-variant: small-caps;
    color: #555;
    letter-spacing: 0.5px;
    margin-bottom: 5px;
    border-bottom: 0.5pt solid #aaa;
    padding-bottom: 2px;
  }}
  h1.title {{
    font-size: 15.5pt;
    font-weight: bold;
    margin: 3px 0 6px 0;
    line-height: 1.20;
  }}
  .authors {{
    font-size: 9.8pt;
    font-weight: bold;
    margin-bottom: 2px;
  }}
  .affiliation {{
    font-size: 8.3pt;
    font-style: italic;
    color: #333;
    margin-bottom: 6px;
  }}
  .meta-note {{
    font-size: 7.4pt;
    color: #444;
    margin-bottom: 10px;
  }}
  .abstract-box {{
    margin: 0 auto 11px auto;
    padding: 7px 11px;
    background: #fbfbfb;
    border: 0.5pt solid #ddd;
    border-radius: 2px;
    font-size: 8.4pt;
    line-height: 1.27;
  }}
  .abstract-box strong.lead {{
    font-variant: small-caps;
    font-weight: bold;
  }}
  .keywords {{
    margin-top: 4px;
    font-size: 7.8pt;
    font-style: italic;
  }}
  .columns {{
    column-count: 2;
    column-gap: 15px;
    column-fill: balance;
  }}
  h2.sec-title {{
    font-size: 9.6pt;
    font-variant: small-caps;
    font-weight: bold;
    border-bottom: 0.5pt solid #222;
    margin: 10px 0 4px 0;
    padding-bottom: 1px;
    text-align: center;
    break-after: avoid;
  }}
  h3.subsec-title {{
    font-size: 9.0pt;
    font-style: italic;
    font-weight: bold;
    margin: 7px 0 2px 0;
    break-after: avoid;
  }}
  p {{
    margin: 0 0 4.5px 0;
    text-indent: 10px;
  }}
  p.no-indent {{
    text-indent: 0;
  }}
  .math-block {{
    text-align: center;
    margin: 4px 0;
    font-style: italic;
    font-size: 8.6pt;
  }}
  table.ieee-tab {{
    width: 100%;
    border-collapse: collapse;
    margin: 5px 0;
    font-size: 7.0pt;
    line-height: 1.18;
    break-inside: avoid;
  }}
  table.ieee-tab th, table.ieee-tab td {{
    padding: 2.2px 3.2px;
    text-align: left;
  }}
  table.ieee-tab th.num, table.ieee-tab td.num {{
    text-align: right;
  }}
  table.ieee-tab th.center, table.ieee-tab td.center {{
    text-align: center;
  }}
  table.ieee-tab thead th {{
    border-top: 1pt solid #000;
    border-bottom: 0.6pt solid #000;
    font-weight: bold;
  }}
  table.ieee-tab tbody tr.rule-bottom td {{
    border-bottom: 1pt solid #000;
  }}
  table.ieee-tab tbody tr.rule-mid td {{
    border-bottom: 0.4pt solid #ccc;
  }}
  .tab-caption {{
    font-size: 7.4pt;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2px;
    font-variant: small-caps;
  }}
  .fig-box {{
    margin: 6px 0;
    text-align: center;
    break-inside: avoid;
  }}
  .fig-box img {{
    max-width: 100%;
    height: auto;
    border: 0.4pt solid #eee;
  }}
  .fig-caption {{
    font-size: 7.3pt;
    margin-top: 2px;
    text-align: justify;
    line-height: 1.18;
  }}
  .full-width {{
    column-span: all;
    margin: 6px 0;
  }}
  .references {{
    font-size: 6.9pt;
    line-height: 1.16;
    margin-top: 6px;
  }}
  .references ol {{
    margin: 0;
    padding-left: 11px;
  }}
  .references li {{
    margin-bottom: 1.8px;
  }}
</style>
</head>
<body>

<div class="header">
  <div class="journal-header">IEEE Journal of Biomedical and Health Informatics / Transactions on Biomedical Engineering, October 2026</div>
  <h1 class="title">Forensic Benchmarking and Schema-Flexible Clinical Tabular Learning: Resolving Data Leakage, Duplicate Contamination, and Inner-Join Collapse in Multi-Cohort Cardiovascular Risk Stratification</h1>
  <div class="authors">Raja Rathan Paida</div>
  <div class="affiliation">Department of Artificial Intelligence, Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat 395007, Gujarat, India</div>
  <div class="meta-note">Correspondence: paida.rajarthan@gmail.com &bull; Open-Source Repository: https://github.com/RathanPaida/SchemaFlexTabNet-QC</div>
</div>

<div class="abstract-box">
  <strong class="lead">Abstract</strong>—Recent literature in cardiovascular risk prediction reports accuracies exceeding 90% on public benchmarks (e.g., 90.1% in Mahmud et al., 2025). In this paper, we conduct a forensic audit of these claims, uncovering critical methodological vulnerabilities in current clinical tabular workflows:
  <strong>(1) Duplicate Contamination and Leakage:</strong> The widely used Kaggle Heart Disease dataset contains 70.5% exact duplicate records (723 of 1,025 rows). Standard random 80/20 train-test splits leak 97.1% of test patients into training, yielding an artificial 98.54% accuracy that plummets to 77.05% upon patient-level deduplication. However, we demonstrate mathematically and empirically that duplicate leakage alone cannot explain the 90.1% claim across the combined 71,025-patient composite (where duplicates constitute only ~1.02% of records); an illustrative fixed-prediction calculation predicts a +0.28% shift, while controlled retraining yields 73.48% (leaked) vs 73.18% (deduplicated). An exhaustive audit reveals internal discrepancies in published confusion matrices (implying 91.35% vs. 89.5% accuracy) and class balance contradictions.
  <strong>(2) Record Overlap and Inner-Join Collapse:</strong> All 270 UCI Statlog records match Cleveland records under a clinical signature, indicating potential duplicate derivation and warranting exclusion against leakage. Naive multi-center merges discard non-overlapping clinical biomarkers (e.g., ST depression, peak heart rate, fluoroscopy), causing zero-shot cross-cohort AUROC to collapse to 0.5353–0.5917.
  To solve these challenges, we introduce <strong>SchemaFlexTabNet-QC</strong>, a tabular transformer framework featuring: (i) a hemodynamic plausibility gatekeeper; (ii) schema-flexible feature tokenization (<em>T<sub>i,j</sub> = <strong>e</strong><sub>name</sub>(f<sub>j</sub>) + &phi;<sub>val</sub>(x<sub>i,j</sub>)</em>) with masked attention and random feature dropout (<em>p<sub>mask</sub> = 0.20</em>), accommodating arbitrary feature subsets without column pruning; (iii) a positive-constrained monotonic residual skip path regularized by autograd penalties (&lambda;<sub>mono</sub> = 1.20), reducing clinical monotonicity violations from 15.04% to 0.26%; (iv) split conformal abstention with Mondrian group-fair calibration; and (v) decentralized Schema-Aware Masked Federated Averaging (Schema-FedAvg).
  Evaluated on 74,013 deduplicated patients across six independent cohorts under strict Leave-One-Cohort-Out (LOCO) testing, SchemaFlexTabNet-QC reaches a mean AUROC of 0.7627 vs. 0.7398 for Union-Schema NaN-aware LightGBM (gain +0.0229); no per-cohort difference is statistically significant after Holm correction, indicating superiority was not established though formal equivalence is not proven. Under conformal abstention (&alpha; = 0.10), zero-shot AUROC rises to <strong>0.8926</strong> (85.15% accuracy, 78.2% acceptance rate) on Hungarian and <strong>0.8755</strong> on Cleveland. Conformal sets calibrated on source cohorts under-cover on held-out hospitals (68.8%–88.4% against a nominal 90%), rigorously bounded by accepted accuracy and falling to 68.8% on Framingham due to outcome construct divergence. Under 50% test-time feature dropout, SchemaFlexTabNet-QC maintains 0.7123 AUROC, retaining an advantage over Native-NaN LightGBM (0.6131) and median-imputed baselines (0.6337). Mondrian calibration halves sex false-positive disparities (0.148 &rarr; 0.072).
  <div class="keywords">
    <strong>Index Terms</strong>—Cardiovascular Risk Prediction, Data Leakage, Forensic Auditing, Tabular Deep Learning, Schema-Flexible Tokenization, Conformal Prediction, Monotonicity Constraints, Federated Learning.
  </div>
</div>

<div class="columns">

<h2 class="sec-title">I. Introduction</h2>
<p>Cardiovascular diseases (CVDs) remain the principal etiology of mortality worldwide, claiming an estimated 17.9 million lives each year [1]. While machine learning holds transformative promise for early risk detection, translating algorithms from single-institution research prototypes to multi-center clinical deployments faces persistent bottlenecks: distribution shift, missing clinical panels, unharmonized electronic health record (EHR) schemas, and evaluation artifacts [15].</p>
<p>In recent years, several published investigations have reported diagnostic accuracies exceeding 90.1% for cardiovascular prediction using deep tabular architectures (notably Mahmud et al., 2025 [2] and related works). However, clinical epidemiologists and translational informatics researchers frequently treat &gt;90% claims on routine clinical vitals with skepticism [3]. On massive, real-world population screening cohorts (e.g., the 70,000-patient Sulianova dataset [2]), the empirical Bayes-optimal performance ceiling established by modern tree ensembles (LightGBM, XGBoost, CatBoost) and deep neural architectures (TabNet, FT-Transformer) hovers consistently between 73.4% and 76.0% accuracy (0.818–0.825 AUROC) [4, 5].</p>
<p>When such performance discrepancies emerge, forensic auditing is essential to determine whether the reported accuracy reflects genuine algorithmic breakthroughs or hidden methodological pitfalls [6]. In this paper, we conduct a comprehensive forensic investigation into recent cardiovascular tabular benchmarks and identify two structural failure modes:</p>
<p class="no-indent">1) <em>Duplicate Contamination and Leakage Fallacy:</em> The popular Kaggle Heart Disease dataset (uploaded by johnsmith88) consists of 1,025 records. However, 723 rows (70.5%) are identical duplicates of other rows. When evaluated with standard random 80/20 train-test splits, 97.1% of test patients are present in the training split, inflating accuracy to an artificial 98.54% (0.9980 AUROC). Upon rigorous patient deduplication, accuracy falls to 77.05% (0.8593 AUROC). Yet, as we demonstrate in Section II, duplicate leakage alone cannot mathematically explain the 90.1% accuracy claimed across the combined 71,025-patient composite (1,025 Kaggle + 70,000 Sulianova rows), because the 723 duplicates represent only ~1.02% of total records. An audit of published manuscripts reveals significant internal reporting contradictions in confusion matrices and class distributions, while rigorous 70/15/15 supervised reproductions achieve an empirical ceiling of 73.48% accuracy.</p>
<p class="no-indent">2) <em>The Destructive Inner-Join Merge:</em> When integrating tertiary cardiology cohorts (e.g., UCI Cleveland, Hungarian, Long Beach VA, Switzerland, and Z-Alizadeh Sani) with large-scale screening databases (Sulianova CVD), conventional clinical data engineering performs an inner-join merge on shared columns. This operation discards diagnostic biomarkers that are not universally collected across all hospitals—specifically ST depression (<code>oldpeak</code>), peak exercise heart rate (<code>thalach</code>), and fluoroscopy vessels (<code>ca</code>). Discarding these predictive features collapses out-of-cohort zero-shot AUROC to 0.5353–0.5917.</p>

<h3 class="subsec-title">A. Key Methodological Contributions</h3>
<p class="no-indent">1) <em>Mathematical and Forensic Audit of the 90.1% Claim:</em> We prove that duplicate contamination leaks 97.1% of test samples in isolated 1,025-row datasets, but demonstrate that across the 71,025-row composite, duplicates represent only ~1.02% of data, which mathematically cannot shift accuracy by +15%. We document internal inconsistencies in published figures (confusion matrix implying 91.35% vs. text implying 89.5%) and report empirical 70/15/15 reproduction benchmarks (73.48% accuracy).</p>
<p class="no-indent">2) <em>Schema-Flexible Tokenization:</em> We formulate a tabular transformer architecture that tokenizes patient records into dynamic feature sets, eliminating column-pruning inner joins and preserving 100% of hospital-specific diagnostic features.</p>
<p class="no-indent">3) <em>Physiological Guarantees via Monotonic Skip Paths:</em> We construct an explicit positive-constrained monotonic skip path combined with autograd gradient penalties (&lambda;<sub>mono</sub> = 1.20), slashing counterfactual monotonicity violations on harmful biomarkers from 15.04% to 0.26%.</p>
<p class="no-indent">4) <em>Conformal Safety and Cohort Shift Analysis:</em> We evaluate split conformal prediction and Mondrian fairness under Leave-One-Cohort-Out (LOCO) testing. We document that cross-cohort distribution shift violates exchangeability, report empirical set coverage alongside acceptance rates, and demonstrate that Mondrian calibration halves sex false-positive disparities (0.148 &rarr; 0.072).</p>
<p class="no-indent">5) <em>Fair DeLong Significance vs. Union-Schema LightGBM:</em> We contrast SchemaFlexTabNet-QC against the proper, union-schema NaN-aware LightGBM baseline with Holm-Bonferroni correction, providing an objective assessment of deep tabular transformers versus gradient-boosted trees.</p>

<h2 class="sec-title">II. Forensic Audit of Baseline Literature</h2>

<h3 class="subsec-title">A. Kaggle Duplicate Contamination (johnsmith88)</h3>
<p>The public Kaggle Heart Disease dataset (<code>heart.csv</code>, uploaded by user johnsmith88) contains <em>N = 1,025</em> patient records across 14 clinical attributes. To investigate the internal composition of this dataset, we executed an exact-match hash audit across all clinical features (<code>age</code>, <code>sex</code>, <code>cp</code>, <code>trestbps</code>, <code>chol</code>, <code>fbs</code>, <code>restecg</code>, <code>thalach</code>, <code>exang</code>, <code>oldpeak</code>, <code>slope</code>, <code>ca</code>, <code>thal</code>, <code>target</code>).</p>
<p>The audit revealed that only 302 unique patient signatures exist within the file. The remaining 723 rows (70.54%) are exact duplicates resulting from non-random sampling with replacement of the original 1989 UCI Cleveland Clinic cohort [7]. When researchers employ standard random 80/20 train/test splits, test records are overwhelmingly drawn from the duplicate pool:</p>
<div class="math-block">P(<strong>x</strong><sub>test</sub> &in; D<sub>train</sub>) = 199 / 205 = 97.07%.</div>
<p>As shown in Table I, an unregularized tree ensemble or neural network evaluated on this split memorizes patient identities, yielding an artificial 98.54% accuracy (0.9980 AUROC). Pruning duplicates and evaluating across unique patients collapses accuracy to 77.05% (0.8593 AUROC).</p>

<h3 class="subsec-title">B. The 90.1% Claim on the 71,025-Patient Composite</h3>
<p>In Mahmud et al. (2025) [2], the authors claim a diagnostic accuracy of 90.1% (AUROC 0.963) on a combined cardiovascular dataset. The combined dataset was constructed by pooling the 1,025-row Kaggle Heart dataset with the 70,000-row Sulianova Cardiovascular dataset [2], totaling <em>N = 71,025</em> rows. The authors specify a 70/15/15 train/validation/test split, yielding a test set of exactly:</p>
<div class="math-block">N<sub>test</sub> = 0.15 &times; 71,025 = 10,653.75 &approx; 10,653 rows.</div>
<p>The exact match of 10,653 test rows confirms that the 1,025-row Kaggle file was pooled into the dataset. However, attributing the 90.1% accuracy solely to memorization of the 723 duplicate rows is mathematically and empirically flawed:</p>
<div class="math-block">Duplicate Fraction = 723 / 71,025 = 1.018% &approx; 1.02%.</div>
<p>In an illustrative fixed-prediction calculation where predictions on non-duplicate records remain invariant, even if every duplicate row were classified with 100% accuracy, raising their accuracy from 73% to 100% would shift overall test accuracy by at most:</p>
<div class="math-block">&Delta; Acc<sub>fixed</sub> &le; 0.0102 &times; (1.00 &minus; 0.73) &approx; +0.28%.</div>
<p>While duplicate retention can also perturb model weights and split compositions, our controlled retraining experiments confirm this order of magnitude: accuracy changes by only +0.30% (73.18% vs. 73.48%). Duplicates therefore cannot explain 90.1% on a pool that is 98.6% Sulianova.</p>
<p>To understand why 90.1% was reported, we conducted a forensic analysis of the published manuscript [2]:</p>
<p class="no-indent">&bull; <em>Confusion Matrix Discrepancy:</em> In Fig. 2 of Mahmud et al., the confusion matrix displays: True Positives (TP) = 5,044, True Negatives (TN) = 4,688, False Positives (FP) = 444, and False Negatives (FN) = 477. The sum of these cells is 5,044 + 4,688 + 444 + 477 = 10,653 (matching the 15% test set). However, computing accuracy directly from this matrix yields:</p>
<div class="math-block">Accuracy = (5,044 + 4,688) / 10,653 = 9,732 / 10,653 = 91.35%,</div>
<p class="no-indent">contradicting the 90.1% stated in the abstract and text.</p>
<p class="no-indent">&bull; <em>Textual Inconsistency:</em> The narrative in Section 4.2 of Mahmud et al. describes a different set of classification counts that yields approximately 89.5% accuracy.</p>
<p class="no-indent">&bull; <em>Class Distribution Discrepancy:</em> In the confusion matrix, disease prevalence is 51.8%, whereas the authors' own data preprocessing section specifies a prevalence filter altering the balance.</p>

<div class="tab-box">
  <div class="tab-caption">Table I: Forensic Audit and Empirical 70/15/15 Reproduction</div>
  <table class="ieee-tab">
    <thead>
      <tr>
        <th>Experimental Setup</th>
        <th class="num">Total N</th>
        <th class="num">Test N</th>
        <th class="num">Accuracy</th>
        <th class="num">AUROC</th>
      </tr>
    </thead>
    <tbody>
      <tr class="rule-mid"><td>Kaggle Leaked Split (80/20)</td><td class="num">1,025</td><td class="num">205</td><td class="num">98.54%</td><td class="num">0.9980</td></tr>
      <tr class="rule-mid"><td>Kaggle Deduplicated Split</td><td class="num">302</td><td class="num">61</td><td class="num">77.05%</td><td class="num">0.8593</td></tr>
      <tr class="rule-mid"><td>Mahmud Reported Baseline</td><td class="num">71,025</td><td class="num">10,653</td><td class="num">90.10%*</td><td class="num">0.9630</td></tr>
      <tr class="rule-mid"><td>Combined Leaked (71,025)</td><td class="num">69,692</td><td class="num">10,454</td><td class="num">73.48%</td><td class="num">0.7978</td></tr>
      <tr class="rule-mid"><td>Combined Deduplicated</td><td class="num">68,969</td><td class="num">10,346</td><td class="num">73.18%</td><td class="num">0.7954</td></tr>
      <tr class="rule-bottom"><td>Sulianova Cohort Alone</td><td class="num">68,667</td><td class="num">10,301</td><td class="num">73.42%</td><td class="num">0.7987</td></tr>
    </tbody>
  </table>
  <div style="font-size: 6.8pt; color: #555; text-align: left; margin-top: 2px;">*Fig. 2 confusion matrix in Mahmud et al. implies 91.35% accuracy; text reports 90.10% and 89.50%.</div>
</div>

<p>To empirically test whether standard supervised learning on this combined schema can produce &gt;90% accuracy, we implemented the 70/15/15 split on the exact combined 71,025-row dataset, the deduplicated dataset, and Sulianova alone under identical hemodynamic filtering (Table I). We explicitly distinguish between (i) a reproduction of the reported data-processing and splitting protocol with standard tabular baselines, (ii) a baseline replication, and (iii) an architecture-faithful reproduction of the proprietary neural model. Using standard LightGBM, Random Forest, and MLP baselines with default hyperparameters and engineered features, the combined leaked pipeline yields <strong>73.48% accuracy</strong> (0.7978 AUROC), the deduplicated pipeline yields <strong>73.18%</strong>, and Sulianova alone yields <strong>73.42%</strong>. We could not reproduce 90.1% with these features and this protocol using standard tabular models. Because original neural weights and splits remain unreleased, we cannot determine whether the original transformer architecture achieved 90.1% through unstated preprocessing or reporting discrepancies.</p>

<h3 class="subsec-title">C. Potential Cross-Cohort Overlap: UCI Cleveland vs. Statlog</h3>
<p>A secondary forensic concern involves evaluating Leave-One-Cohort-Out transfer on UCI Statlog (<em>N=270</em>) after training on UCI Cleveland (<em>N=303</em>). A biometric signature audit across six core physiological parameters (age, sex, resting BP, cholesterol, peak heart rate, ST depression) revealed that all 270 Statlog records match Cleveland records under this clinical-feature signature. While matching clinical measurements alone cannot conclusively verify unique biological patient identity without institutional identifiers, this exact match strongly indicates record-level overlap or duplicate derivation from the same underlying registry. Treating these records as overlapping is a necessary precaution against data leakage, and all 270 Statlog rows were excluded from our cross-cohort experiments.</p>

<h3 class="subsec-title">D. The Destructive Inner-Join Merge</h3>
<p>In multi-center healthcare systems, different clinical sites collect distinct lab panels. When merging datasets, standard pipelines perform an inner join, discarding unshared columns. In cardiovascular care, screening datasets (e.g., Sulianova) collect non-invasive vitals (blood pressure, cholesterol, glucose, BMI), while tertiary cardiac catheterization labs (e.g., UCI cohorts, Z-Alizadeh Sani) measure coronary fluoroscopy (<code>ca</code>), exercise ST depression (<code>oldpeak</code>), and peak heart rate (<code>thalach</code>).</p>
<p>As detailed in Table III, performing an inner join forces tertiary models to rely solely on non-invasive vitals. When evaluated zero-shot on held-out hospitals, inner-join LightGBM collapses to 0.5917 AUROC on Hungarian and 0.5353 on VA + Switzerland. Discarding specialized diagnostic columns severely impairs zero-shot clinical utility.</p>

<div class="fig-box">
  <img src="{fig1_b64}" alt="Figure 1: Forensic Audit Overview">
  <div class="fig-caption"><strong>Fig. 1.</strong> Forensic audit of literature benchmarks: (Left) Distribution of 723 duplicate rows in Kaggle Heart. (Middle) Artificial performance jump under data leakage vs. deduplicated benchmark. (Right) Zero-shot performance collapse under inner joins.</div>
</div>

<h2 class="sec-title">III. Proposed Architecture: SchemaFlexTabNet-QC</h2>

<h3 class="subsec-title">A. Hemodynamic Plausibility Gatekeeper</h3>
<p>EHR databases contain uncalibrated cuff artifacts and sensor transcription errors. Prior to tokenization, input vitals pass through a physiological gatekeeper based on human hemodynamic limits:</p>
<div class="math-block">SBP &in; [60, 260] mmHg, &nbsp; DBP &in; [40, 160] mmHg,</div>
<div class="math-block">PP = SBP &minus; DBP &ge; 10 mmHg.</div>
<p>Records violating these physical constraints (e.g., inverted SBP &lt; DBP) are filtered. In Sulianova, this identifies 1,374 physiologically corrupt rows.</p>

<h3 class="subsec-title">B. Schema-Flexible Feature Tokenization</h3>
<p>Let a patient <em>i</em> be characterized by an arbitrary observed feature subset <em>F<sub>i</sub> &sube; {{f<sub>1</sub>, f<sub>2</sub>, ..., f<sub>M</sub>}}</em> drawn from the global union of clinical attributes (<em>M=22</em>). Rather than arranging inputs into a static matrix with imputed zeros, each observed feature <em>j &in; F<sub>i</sub></em> is transformed into a continuous vector token <em>T<sub>i,j</sub> &in; R<sup>d</sup></em>:</p>
<div class="math-block">T<sub>i,j</sub> = <strong>e</strong><sub>name</sub>(f<sub>j</sub>) + &phi;<sub>val</sub>(x<sub>i,j</sub>),</div>
<p>where <strong>e</strong><sub>name</sub>(f<sub>j</sub>) &in; R<sup>d</sup> is a learned feature-identity embedding, and &phi;<sub>val</sub>(x<sub>i,j</sub>) is a feature-specific linear projection:</p>
<div class="math-block">&phi;<sub>val</sub>(x<sub>i,j</sub>) = x<sub>i,j</sub> &middot; <strong>w</strong><sub>j</sub> + <strong>b</strong><sub>j</sub>, &nbsp; <strong>w</strong><sub>j</sub>, <strong>b</strong><sub>j</sub> &in; R<sup>d</sup>.</div>
<p>For numerical stability, continuous variables are z-score standardized using training statistics. Prepending a learned [CLS] &in; R<sup>d</sup> token yields a variable-length sequence:</p>
<div class="math-block"><strong>Z</strong><sub>0</sub> = [ [CLS], T<sub>i,j<sub>1</sub></sub>, T<sub>i,j<sub>2</sub></sub>, ..., T<sub>i,j<sub>|F<sub>i</sub>|</sub></sub> ] &in; R<sup>(1 + |F<sub>i</sub>|) &times; d</sup>.</div>
<p>The sequence is processed by an <em>L</em>-layer Transformer Encoder (<em>L=2</em>, <em>H=4</em> attention heads, <em>d=64</em>, <em>d<sub>ff</sub>=128</em>, dropout 0.10):</p>
<div class="math-block"><strong>Z</strong>'&subl; = LayerNorm(<strong>Z</strong><sub>l-1</sub> + MHA(<strong>Z</strong><sub>l-1</sub>, <strong>M</strong><sub>i</sub>)),</div>
<div class="math-block"><strong>Z</strong>&subl; = LayerNorm(<strong>Z</strong>'&subl; + FFN(<strong>Z</strong>'&subl;)),</div>
<p>where <strong>M</strong><sub>i</sub> is a boolean key padding mask ensuring unobserved tokens receive zero attention. To instill missing-schema robustness, we inject random feature dropout during training with probability <em>p<sub>mask</sub> = 0.20</em>.</p>

<h3 class="subsec-title">C. Positive-Constrained Monotonic Skip Path</h3>
<p>Cardiovascular clinicians require that established harmful biomarkers (e.g., advancing age, malignant systolic blood pressure, high cholesterol, severe ST depression) must not lower predicted disease risk when elevated counterfactually [15]. Because unconstrained multi-head attention can produce non-monotonic artifacts, we introduce a dual inductive bias:</p>
<p class="no-indent">1) <em>Positive-Constrained Skip Path:</em> We define a monotonic linear skip branch for harmful biomarkers <em>M = {{age, ap_hi, trestbps, chol, oldpeak}}</em>:</p>
<div class="math-block">f<sub>mono</sub>(<strong>x</strong><sub>i</sub>) = &sum;<sub>k &in; M</sub> softplus(w<sub>k</sub>) &middot; x<sub>i,k</sub> &middot; I(x<sub>i,k</sub> observed),</div>
<p>where softplus(w) = ln(1 + e<sup>w</sup>) &gt; 0 strictly enforces non-negative coefficients.</p>
<p class="no-indent">2) <em>Autograd Monotonicity Regularization:</em> We compute the exact input gradient via automatic differentiation and penalize negative directional gradients:</p>
<div class="math-block">L<sub>mono</sub> = (1 / |M|) &sum;<sub>k &in; M</sub> ReLU( &minus; &part;y / &part;x<sub>i,k</sub> ) &middot; I(x<sub>i,k</sub> observed).</div>
<p>The final logit is the sum of the transformer [CLS] projection and the monotonic branch:</p>
<div class="math-block">y&#770;<sub>i</sub> = &sigma;( <strong>W</strong><sub>out</sub> <strong>z</strong><sub>CLS</sub> + b<sub>0</sub> + f<sub>mono</sub>(<strong>x</strong><sub>i</sub>) ).</div>
<p>The total loss function is:</p>
<div class="math-block">L<sub>total</sub> = L<sub>BCE</sub>(y&#770;<sub>i</sub>, y<sub>i</sub>) + &lambda;<sub>1</sub> L<sub>concept</sub> + &lambda;<sub>2</sub> L<sub>mono</sub>,</div>
<p>where L<sub>concept</sub> is an auxiliary concept bottleneck predicting Stage 2 Hypertension (SBP &ge; 140 or DBP &ge; 90 mmHg) with &lambda;<sub>1</sub> = 0.10, and &lambda;<sub>2</sub> = 1.20.</p>

<h2 class="sec-title">IV. Conformal Prediction and Fairness Under Cohort Shift</h2>

<h3 class="subsec-title">A. Conformal Abstention and Exchangeability Violations</h3>
<p>In Split Conformal Prediction [8, 9], non-conformity scores <em>s<sub>i</sub> = 1 &minus; P&#770;(Y<sub>i</sub> = y<sub>i</sub> | <strong>X</strong><sub>i</sub>)</em> are computed on a calibration split D<sub>cal</sub>. Given an error tolerance &alpha; &in; (0, 1), the conformal quantile is:</p>
<div class="math-block">q&#770; = Quantile( &lceil; (n<sub>cal</sub> + 1)(1 &minus; &alpha;) &rceil; / n<sub>cal</sub>; {{s<sub>i</sub>}} ).</div>
<p>For a new patient <strong>X</strong><sub>n+1</sub>, the prediction set is constructed as:</p>
<div class="math-block">C(<strong>X</strong><sub>n+1</sub>) = {{ y &in; {{0, 1}} | 1 &minus; P&#770;(Y = y | <strong>X</strong><sub>n+1</sub>) &le; q&#770; }}.</div>
<p>When calibration and testing occur on independent, identically distributed samples from the same cohort, exchangeability guarantees marginal set coverage:</p>
<div class="math-block">P(Y<sub>n+1</sub> &in; C(<strong>X</strong><sub>n+1</sub>)) &ge; 1 &minus; &alpha;.</div>
<p>However, in Leave-One-Cohort-Out clinical deployments, calibration is performed on source cohorts while testing occurs on a distinct held-out hospital. Under this cross-cohort shift:</p>
<div class="math-block">P<sub>source</sub>(<strong>X</strong>, Y) &ne; P<sub>target</sub>(<strong>X</strong>, Y).</div>
<p>Exchangeability is mathematically violated [11, 12]. As demonstrated in Table IV, empirical target set coverage drops below nominal 1 &minus; &alpha; = 0.90 (ranging from 84.2% on Z-Alizadeh Sani to 88.7% on Hungarian). Calibrating on a small held-out calibration partition from the target cohort restores exact coverage (&ge; 90.1%).</p>
<p>In our clinical reporting, we strictly distinguish between:</p>
<p class="no-indent">&bull; <em>Conformal Set Coverage:</em> The probability that the true label belongs to the prediction set, P(Y &in; C(X)).</p>
<p class="no-indent">&bull; <em>Acceptance Rate (Selection Rate):</em> The fraction of decisive singleton predictions, |C(X)| = 1, where the model does not abstain.</p>

<h3 class="subsec-title">B. Mondrian Group-Fair Conformal Calibration</h3>
<p>To mitigate disparities across demographic groups <em>G &in; {{Male, Female}}</em>, standard global thresholding can yield unequal false-positive rates. Mondrian conformal calibration computes group-conditional quantiles q&#770;<sub>g</sub> [10]:</p>
<div class="math-block">P(Y<sub>n+1</sub> &in; C(<strong>X</strong><sub>n+1</sub>) | G<sub>n+1</sub> = g) &ge; 1 &minus; &alpha;, &nbsp; &forall; g &in; {{M, F}}.</div>
<p>By calibrating subgroup decision thresholds against equalized operating targets on the calibration split, Mondrian calibration substantially mitigates disparities across sexes.</p>

<h3 class="subsec-title">C. Schema-Aware Federated Learning (Schema-FedAvg)</h3>
<p>In multi-hospital consortia, patient records cannot leave local firewalls. When hospitals possess non-identical schemas, naive FedAvg [14] fails because clients update disjoint weight parameters. We implement <strong>Schema-Aware Masked FedAvg</strong>: each hospital <em>k</em> trains locally on its observed token subset F<sup>(k)</sup>. During central aggregation, global parameter tensors are updated using presence-masked weighting:</p>
<div class="math-block"><strong>W</strong><sub>t+1</sub>[j] = &sum;<sub>k &in; S<sub>j</sub></sub> n<sub>k</sub> <strong>W</strong><sub>t,k</sub>[j] / &sum;<sub>k &in; S<sub>j</sub></sub> n<sub>k</sub>,</div>
<p>where S<sub>j</sub> = {{k | feature j &in; F<sup>(k)</sup>}}. Hospitals collaborate without transmitting raw patient rows or imputing missing columns.</p>

<h2 class="sec-title">V. Experimental Setup</h2>

<h3 class="subsec-title">A. Multi-Cohort Harmonization</h3>
<p>We assemble and harmonize six independent cardiovascular cohorts spanning 74,013 clean, non-overlapping patients:</p>
<p class="no-indent">1) <em>Sulianova Population Screening Cohort (N = 68,568 clean):</em> Primary screening cohort in Russia recording demographics, blood pressure, cholesterol, glucose, and lifestyle vitals. Disease endpoint: prevalent cardiovascular disease survey.</p>
<p class="no-indent">2) <em>UCI Cleveland Clinic (N = 303 clean):</em> Tertiary cardiac catheterization cohort. Disease endpoint: angiographic coronary artery disease (&ge; 50% diameter stenosis) [7].</p>
<p class="no-indent">3) <em>UCI Hungarian Institute of Cardiology (N = 293):</em> Angiographic cohort in Budapest [18].</p>
<p class="no-indent">4) <em>UCI Long Beach VA &amp; Switzerland (N = 308 pooled):</em> High-risk veterans and Swiss cardiology patients [19].</p>
<p class="no-indent">5) <em>Z-Alizadeh Sani Cohort (N = 303):</em> Tertiary catheterization registry from Tehran Heart Center [16].</p>
<p class="no-indent">6) <em>Framingham Heart Study (N = 4,238):</em> Longitudinal epidemiological cohort [17]. Disease endpoint: 10-year incident coronary heart disease event.</p>

<h3 class="subsec-title">B. Training Hyperparameters</h3>
<p>Models are trained with PyTorch on NVIDIA GPUs using AdamW optimizer (lr = 1.5 &times; 10<sup>-3</sup>, weight decay = 1 &times; 10<sup>-4</sup>), batch size 256, for 16 epochs with cosine annealing schedule. Random seeds are fixed (Seed 42). Gradient clipping is applied at max norm 1.0.</p>

<h2 class="sec-title">VI. Results and Empirical Analysis</h2>

<h3 class="subsec-title">A. Baseline Multi-Model Benchmarking (Table II)</h3>
<p>We benchmark seven competitive tabular architectures across Stratified 5-Fold Cross-Validation on the harmonized multi-cohort dataset (<em>N=74,013</em>, Statlog duplicates excluded). Configuration A evaluates raw features (10 features); Configuration B includes engineered physiological markers (BMI, pulse pressure, ATP-III tiers).</p>
<p>As presented in Table II, all top-performing models saturate within a narrow performance band: LightGBM achieves 75.39% accuracy (0.8187 AUROC) on raw features and 75.98% accuracy (0.8249 AUROC) with meta-features. Random Forest (75.85%), XGBoost (75.95%), and CatBoost (75.60%) demonstrate near-identical metrics. Deep tabular baselines (TabNet, MLP) achieve 74.87% accuracy (0.8105 AUROC). This confirms that on noisy population screening data, the empirical accuracy ceiling resides at approximately 75.4%–76.0%.</p>

<div class="tab-box">
  <div class="tab-caption">Table II: Baseline Benchmarking Across Tabular Architectures (5-Fold CV, N=74,013)</div>
  <table class="ieee-tab">
    <thead>
      <tr>
        <th>Architecture</th>
        <th class="num">Raw AUROC</th>
        <th class="num">Raw Acc</th>
        <th class="num">Meta AUROC</th>
        <th class="num">Meta Acc</th>
      </tr>
    </thead>
    <tbody>
      <tr class="rule-mid"><td>Logistic Regression</td><td class="num">0.7554</td><td class="num">69.16%</td><td class="num">0.7678</td><td class="num">70.19%</td></tr>
      <tr class="rule-mid"><td>Random Forest</td><td class="num">0.8195</td><td class="num">75.03%</td><td class="num">0.8244</td><td class="num">75.85%</td></tr>
      <tr class="rule-mid"><td>XGBoost</td><td class="num">0.8186</td><td class="num">75.27%</td><td class="num">0.8247</td><td class="num">75.95%</td></tr>
      <tr class="rule-mid"><td><strong>LightGBM</strong></td><td class="num"><strong>0.8187</strong></td><td class="num"><strong>75.39%</strong></td><td class="num"><strong>0.8249</strong></td><td class="num"><strong>75.98%</strong></td></tr>
      <tr class="rule-mid"><td>CatBoost</td><td class="num">0.8152</td><td class="num">75.13%</td><td class="num">0.8215</td><td class="num">75.60%</td></tr>
      <tr class="rule-mid"><td>MLP Neural Net</td><td class="num">0.8096</td><td class="num">74.34%</td><td class="num">0.8110</td><td class="num">74.86%</td></tr>
      <tr class="rule-bottom"><td>TabNet</td><td class="num">0.8078</td><td class="num">74.53%</td><td class="num">0.8105</td><td class="num">74.87%</td></tr>
    </tbody>
  </table>
</div>

<h3 class="subsec-title">B. Multi-Cohort Leave-One-Cohort-Out Matrix (Table III)</h3>
<p>Table III presents the primary out-of-cohort generalization benchmark across five held-out evaluation targets. In each experiment, models are trained on all other cohorts and evaluated zero-shot on the target cohort.</p>

<div class="tab-box">
  <div class="tab-caption">Table III: Multi-Cohort Leave-One-Cohort-Out (LOCO) Evaluation Matrix</div>
  <table class="ieee-tab">
    <thead>
      <tr>
        <th>Held-Out Target Cohort</th>
        <th>Model Architecture</th>
        <th class="num">Accept. Rate</th>
        <th class="num">AUROC</th>
        <th class="num">Accuracy</th>
        <th class="num">ECE</th>
      </tr>
    </thead>
    <tbody>
      <tr class="rule-mid"><td rowspan="4">UCI Hungarian<br>(N=293)</td><td>Inner-Join LightGBM</td><td class="num">100.0%</td><td class="num">0.5917</td><td class="num">55.63%</td><td class="num">0.2160</td></tr>
      <tr class="rule-mid"><td>Union-Schema LightGBM</td><td class="num">100.0%</td><td class="num">0.8423</td><td class="num">74.06%</td><td class="num">0.1325</td></tr>
      <tr class="rule-mid"><td>HybridTabNet-QC Baseline</td><td class="num">100.0%</td><td class="num">0.6111</td><td class="num">59.04%</td><td class="num">0.1254</td></tr>
      <tr class="rule-mid"><td><strong>SchemaFlex (Full)</strong></td><td class="num">100.0%</td><td class="num"><strong>0.8641</strong></td><td class="num"><strong>76.79%</strong></td><td class="num"><strong>0.1026</strong></td></tr>
      <tr class="rule-mid"><td><strong>SchemaFlex (Conformal)</strong></td><td class="num"><strong>78.2%</strong></td><td class="num"><strong>0.8926</strong></td><td class="num"><strong>85.15%</strong></td><td class="num"><strong>0.0756</strong></td></tr>
      <tr class="rule-mid"><td rowspan="4">UCI VA + Switz.<br>(N=308)</td><td>Inner-Join LightGBM</td><td class="num">100.0%</td><td class="num">0.5353</td><td class="num">53.57%</td><td class="num">0.3047</td></tr>
      <tr class="rule-mid"><td>Union-Schema LightGBM</td><td class="num">100.0%</td><td class="num"><strong>0.7280</strong></td><td class="num">72.40%</td><td class="num">0.2134</td></tr>
      <tr class="rule-mid"><td>HybridTabNet-QC Baseline</td><td class="num">100.0%</td><td class="num">0.5451</td><td class="num">70.13%</td><td class="num">0.1921</td></tr>
      <tr class="rule-mid"><td><strong>SchemaFlex (Full)</strong></td><td class="num">100.0%</td><td class="num">0.7241</td><td class="num"><strong>74.03%</strong></td><td class="num"><strong>0.1382</strong></td></tr>
      <tr class="rule-mid"><td><strong>SchemaFlex (Conformal)</strong></td><td class="num"><strong>72.7%</strong></td><td class="num"><strong>0.7705</strong></td><td class="num"><strong>83.48%</strong></td><td class="num"><strong>0.0623</strong></td></tr>
      <tr class="rule-mid"><td rowspan="4">UCI Cleveland<br>(N=303)</td><td>Inner-Join LightGBM</td><td class="num">100.0%</td><td class="num">0.5705</td><td class="num">56.44%</td><td class="num">0.1759</td></tr>
      <tr class="rule-mid"><td>Union-Schema LightGBM</td><td class="num">100.0%</td><td class="num"><strong>0.8566</strong></td><td class="num"><strong>76.90%</strong></td><td class="num"><strong>0.0559</strong></td></tr>
      <tr class="rule-mid"><td>HybridTabNet-QC Baseline</td><td class="num">100.0%</td><td class="num">0.6785</td><td class="num">62.71%</td><td class="num">0.0890</td></tr>
      <tr class="rule-mid"><td><strong>SchemaFlex (Full)</strong></td><td class="num">100.0%</td><td class="num">0.8433</td><td class="num">73.60%</td><td class="num">0.1209</td></tr>
      <tr class="rule-mid"><td><strong>SchemaFlex (Conformal)</strong></td><td class="num"><strong>78.9%</strong></td><td class="num"><strong>0.8755</strong></td><td class="num"><strong>78.66%</strong></td><td class="num"><strong>0.0946</strong></td></tr>
      <tr class="rule-mid"><td rowspan="2">Z-Alizadeh Sani<br>(N=303)</td><td>Union-Schema LightGBM</td><td class="num">100.0%</td><td class="num">0.6028</td><td class="num">68.98%</td><td class="num">0.1148</td></tr>
      <tr class="rule-mid"><td><strong>SchemaFlex (Full)</strong></td><td class="num">100.0%</td><td class="num"><strong>0.7043</strong></td><td class="num">62.71%</td><td class="num">0.2072</td></tr>
      <tr class="rule-mid"><td><strong>SchemaFlex (Conformal)</strong></td><td class="num"><strong>36.6%</strong></td><td class="num"><strong>0.7895</strong></td><td class="num"><strong>72.07%</strong></td><td class="num">0.1921</td></tr>
      <tr class="rule-mid"><td rowspan="2">Framingham Study<br>(N=4,238)</td><td>Union-Schema LightGBM</td><td class="num">100.0%</td><td class="num">0.6695</td><td class="num"><strong>53.26%</strong></td><td class="num">0.3614</td></tr>
      <tr class="rule-bottom"><td><strong>SchemaFlex (Full)</strong></td><td class="num">100.0%</td><td class="num"><strong>0.6778</strong></td><td class="num">51.32%</td><td class="num">0.3958</td></tr>
    </tbody>
  </table>
</div>

<p>On UCI Hungarian, SchemaFlexTabNet-QC achieves 0.8641 AUROC (76.79% accuracy), significantly surpassing Inner-Join LightGBM (0.5917) and outperforming Union-Schema LightGBM (0.8423). Under Conformal Abstention (&alpha; = 0.10), the acceptance rate is 78.2%, and AUROC on the accepted cohort rises to <strong>0.8926</strong> (85.15% accuracy).</p>
<p>On UCI Cleveland, Union-Schema LightGBM achieves 0.8566 AUROC vs. SchemaFlexTabNet-QC's 0.8433 (delta &minus;0.0133), demonstrating that gradient-boosted trees remain formidable baselines when trained on union schemas. However, conformal abstention on SchemaFlexTabNet-QC lifts AUROC to <strong>0.8755</strong> (78.66% accuracy).</p>
<p>On Z-Alizadeh Sani, SchemaFlexTabNet-QC markedly outperforms Union-Schema LightGBM (0.7043 vs. 0.6028, gain +0.1015).</p>

<h3 class="subsec-title">C. Rigorous DeLong Significance Testing (Table IV)</h3>
<p>Previous reports compared deep models against an inner-join LightGBM baseline, reporting extreme significance (<em>p &approx; 10<sup>-14</sup></em>). To eliminate straw-man comparisons, Table IV evaluates paired DeLong statistical significance tests [13] against the true competitor: <strong>Union-Schema NaN-Aware LightGBM</strong>, with Holm-Bonferroni step-down correction across all five cohorts.</p>

<div class="tab-box">
  <div class="tab-caption">Table IV: Paired DeLong Tests vs. Baselines Across Held-Out Cohorts</div>
  <table class="ieee-tab">
    <thead>
      <tr>
        <th>Held-Out Cohort</th>
        <th class="num">Proposed</th>
        <th class="num">Union LGB</th>
        <th class="num">&Delta;</th>
        <th class="num">Raw <em>p</em></th>
        <th class="num">Holm <em>p</em></th>
      </tr>
    </thead>
    <tbody>
      <tr class="rule-mid"><td>UCI Hungarian</td><td class="num">0.8641</td><td class="num">0.8423</td><td class="num">+0.0218</td><td class="num">0.1578</td><td class="num">0.6310 (n.s.)</td></tr>
      <tr class="rule-mid"><td>UCI VA + Switz.</td><td class="num">0.7241</td><td class="num">0.7280</td><td class="num">-0.0039</td><td class="num">0.7984</td><td class="num">1.0000 (n.s.)</td></tr>
      <tr class="rule-mid"><td>UCI Cleveland</td><td class="num">0.8433</td><td class="num">0.8566</td><td class="num">-0.0133</td><td class="num">0.4472</td><td class="num">1.0000 (n.s.)</td></tr>
      <tr class="rule-mid"><td>Z-Alizadeh Sani</td><td class="num">0.7043</td><td class="num">0.6028</td><td class="num">+0.1015</td><td class="num">0.0369</td><td class="num">0.1846 (n.s.)</td></tr>
      <tr class="rule-mid"><td>Framingham Study</td><td class="num">0.6778</td><td class="num">0.6695</td><td class="num">+0.0083</td><td class="num">0.4700</td><td class="num">1.0000 (n.s.)</td></tr>
      <tr class="rule-mid"><td colspan="6" style="font-style: italic; background: #fafafa;">Comparison vs. HybridTabNet-QC (Inner-Join Baseline):</td></tr>
      <tr class="rule-mid"><td>UCI Hungarian</td><td class="num">0.8641</td><td class="num">0.6111</td><td class="num">+0.2530</td><td class="num">&lt; 10<sup>-10</sup></td><td class="num">&lt; 10<sup>-10</sup> (***)</td></tr>
      <tr class="rule-mid"><td>UCI VA + Switz.</td><td class="num">0.7241</td><td class="num">0.5451</td><td class="num">+0.1790</td><td class="num">5.6 &times; 10<sup>-4</sup></td><td class="num">1.7 &times; 10<sup>-3</sup> (**)</td></tr>
      <tr class="rule-bottom"><td>UCI Cleveland</td><td class="num">0.8433</td><td class="num">0.6785</td><td class="num">+0.1648</td><td class="num">2.1 &times; 10<sup>-6</sup></td><td class="num">8.4 &times; 10<sup>-6</sup> (***)</td></tr>
    </tbody>
  </table>
</div>

<p>Against the inner-join HybridTabNet-QC, SchemaFlexTabNet-QC is overwhelmingly superior (<em>p &lt; 10<sup>-5</sup></em> to <em>p &lt; 10<sup>-10</sup></em>). However, against Union-Schema LightGBM, the mean AUROC gain across the five cohorts is modest (+0.0229). We did not detect a statistically significant AUROC difference between SchemaFlexTabNet-QC and Union-Schema LightGBM on any individual cohort after Holm-Bonferroni correction. While this establishes that superiority was not demonstrated under the specified tests, it does not prove statistical equivalence. Given the modest sample sizes (<em>N &approx; 300</em>) of the tertiary cardiology cohorts, statistical power is limited to detect small margins (&Delta; &approx; 0.01–0.02); establishing practical equivalence requires prespecified equivalence margins and formal equivalence testing on larger cohorts.</p>

<div class="fig-box">
  <img src="{fig2_b64}" alt="Figure 2: Risk Coverage Tradeoff">
  <div class="fig-caption"><strong>Fig. 2.</strong> Conformal risk-coverage profile: Diagnostic accuracy and AUROC scale gracefully as ambiguous predictions are deferred for clinician consultation.</div>
</div>

<h3 class="subsec-title">D. Conformal Cohort Shift, Mathematical Upper Bound, and Underlying Count Audit (Table V)</h3>
<p>For binary classification on <em>N</em> evaluation patients, prediction sets satisfy |<em>C(x<sub>i</sub>)</em>| &in; {{0, 1, 2}}. Let <em>N</em><sub>0</sub> denote empty sets, <em>N</em><sub>1</sub> singleton sets (accepted predictions; <em>r</em><sub>acc</sub> = <em>N</em><sub>1</sub>/<em>N</em>), and <em>N</em><sub>2</sub> uncertain pair sets ({0, 1}), with <em>N</em> = <em>N</em><sub>0</sub> + <em>N</em><sub>1</sub> + <em>N</em><sub>2</sub>. Accepted patients achieve accuracy Acc<sub>acc</sub> = <em>N</em><sub>1,corr</sub>/<em>N</em><sub>1</sub>. Crucially, for every incorrectly predicted accepted patient, <em>C(x<sub>i</sub>)</em> = {{1 &minus; <em>y<sub>i</sub></em>}}, so the true label <em>y<sub>i</sub></em> is strictly absent. Empty sets likewise contain no labels. Therefore, true label coverage <em>y<sub>i</sub> &in; C(x<sub>i</sub>)</em> is satisfied only by correct singletons and pair sets: <em>N</em><sub>cov</sub> = <em>N</em><sub>1,corr</sub> + <em>N</em><sub>2</sub> = <em>N</em> &minus; <em>N</em><sub>0</sub> &minus; <em>N</em><sub>1</sub>(1 &minus; Acc<sub>acc</sub>). Dividing by <em>N</em> yields the exact coverage identity and theoretical upper bound:</p>
<div class="math-block">Coverage = <em>N</em><sub>cov</sub> / <em>N</em> = 1 &minus; <em>N</em><sub>0</sub>/<em>N</em> &minus; <em>r</em><sub>acc</sub>(1 &minus; Acc<sub>acc</sub>) &le; 1 &minus; <em>r</em><sub>acc</sub>(1 &minus; Acc<sub>acc</sub>)</div>
<p>Table V provides the complete verified count audit reconciling accepted accuracy and empirical set coverage. For Hungarian (<em>N</em>=293), with 229 singletons (195 correct, 34 errors, 85.2% accuracy) and 64 uncertain pairs (<em>N</em><sub>0</sub>=0), exact empirical coverage is 259/293 = <strong>88.4%</strong>, matching the theoretical upper bound 1 &minus; 0.7816 &times; (1 &minus; 0.8515) = 88.4%. For Cleveland (<em>N</em>=303), 239 singletons (188 correct, 51 errors) and 64 pairs yield <em>N</em><sub>cov</sub>=252, giving empirical coverage <strong>83.2%</strong>, matching the theoretical ceiling 83.2%. For Framingham (<em>N</em>=4,238), where outcome construct divergence drops accepted accuracy to 51.2% across 2,715 singletons (1,324 errors), with 1,523 uncertain pairs and <em>N</em><sub>0</sub>=0, empirical coverage is 2,914/4,238 = <strong>68.8%</strong>, bounded by 68.8%. For VA+Switzerland (<em>N</em>=308, <em>N</em><sub>0</sub>=5) and Z-Alizadeh Sani (<em>N</em>=303, <em>N</em><sub>0</sub>=17), small numbers of empty sets yield coverages of 86.4% and 84.2%, strictly below their theoretical ceilings of 88.0% and 89.8%. This fully reconciles earlier draft discrepancies and proves that cross-cohort shift degrades marginal coverage (68.8%–88.4% vs. nominal 90%).</p>

<div class="tab-box">
  <div class="tab-caption">Table V: Reconciled Patient-Level Conformal Count Audit and Theoretical Upper Bounds</div>
  <table class="ieee-tab">
    <thead>
      <tr>
        <th>Target Cohort</th>
        <th class="num">Test <em>N</em></th>
        <th class="num"><em>N</em><sub>1</sub> (<em>r</em><sub>acc</sub>)</th>
        <th class="num">Acc. (%)</th>
        <th class="num"><em>N</em><sub>2</sub></th>
        <th class="num"><em>N</em><sub>0</sub></th>
        <th class="num"><em>N</em><sub>cov</sub></th>
        <th class="num">Src. Cov.</th>
        <th class="num">Max Bound</th>
        <th class="num">Tgt. Cov.</th>
      </tr>
    </thead>
    <tbody>
      <tr class="rule-mid"><td>UCI Hungarian</td><td class="num">293</td><td class="num">229 (78.2%)</td><td class="num">85.2%</td><td class="num">64</td><td class="num">0</td><td class="num">259</td><td class="num"><strong>88.4%</strong></td><td class="num">88.4%</td><td class="num">90.4%</td></tr>
      <tr class="rule-mid"><td>UCI VA + Switz.</td><td class="num">308</td><td class="num">224 (72.7%)</td><td class="num">83.5%</td><td class="num">79</td><td class="num">5</td><td class="num">266</td><td class="num"><strong>86.4%</strong></td><td class="num">88.0%</td><td class="num">90.3%</td></tr>
      <tr class="rule-mid"><td>UCI Cleveland</td><td class="num">303</td><td class="num">239 (78.9%)</td><td class="num">78.7%</td><td class="num">64</td><td class="num">0</td><td class="num">252</td><td class="num"><strong>83.2%</strong></td><td class="num">83.2%</td><td class="num">90.1%</td></tr>
      <tr class="rule-mid"><td>Z-Alizadeh Sani</td><td class="num">303</td><td class="num">111 (36.6%)</td><td class="num">72.1%</td><td class="num">175</td><td class="num">17</td><td class="num">255</td><td class="num"><strong>84.2%</strong></td><td class="num">89.8%</td><td class="num">90.1%</td></tr>
      <tr class="rule-bottom"><td>Framingham Study</td><td class="num">4,238</td><td class="num">2,715 (64.1%)</td><td class="num">51.2%</td><td class="num">1,523</td><td class="num">0</td><td class="num">2,914</td><td class="num"><strong>68.8%</strong></td><td class="num">68.8%</td><td class="num">90.0%</td></tr>
    </tbody>
  </table>
</div>

<p>Crucially, on the Framingham cohort, conformal abstention does not improve classification accuracy (51.32% full vs. 51.23% accepted, &Delta; = &minus;0.09%). Because Framingham tracks 10-year incident events rather than cross-sectional angiographic CAD, the prediction score ranking is less reliable, causing abstention to discard both correct and incorrect predictions equally.</p>

<h3 class="subsec-title">E. Test-Time Missingness vs. Native-NaN Trees (Table VI)</h3>
<p>Prior studies claimed a +0.15 AUROC advantage over tree models under test-time missingness, but evaluated against median-imputed trees. In Table VI, we evaluate SchemaFlexTabNet-QC against <strong>Native-NaN LightGBM</strong> and <strong>Native-NaN XGBoost</strong>, which natively route missing values through optimal split paths.</p>

<div class="tab-box">
  <div class="tab-caption">Table VI: Test-Time Missingness Stress Test Under Feature Dropout</div>
  <table class="ieee-tab">
    <thead>
      <tr>
        <th>Missing Rate</th>
        <th class="num">SchemaFlex</th>
        <th class="num">Native LGB</th>
        <th class="num">Native XGB</th>
        <th class="num">Med. LGB</th>
        <th class="num">&Delta; Native</th>
      </tr>
    </thead>
    <tbody>
      <tr class="rule-mid"><td>0%</td><td class="num"><strong>0.8520</strong></td><td class="num">0.7297</td><td class="num">0.7265</td><td class="num">0.7274</td><td class="num">+0.1223</td></tr>
      <tr class="rule-mid"><td>10%</td><td class="num"><strong>0.8300</strong></td><td class="num">0.7035</td><td class="num">0.7051</td><td class="num">0.7090</td><td class="num">+0.1265</td></tr>
      <tr class="rule-mid"><td>20%</td><td class="num"><strong>0.8036</strong></td><td class="num">0.6854</td><td class="num">0.6940</td><td class="num">0.6944</td><td class="num">+0.1182</td></tr>
      <tr class="rule-mid"><td>30%</td><td class="num"><strong>0.7510</strong></td><td class="num">0.6718</td><td class="num">0.6822</td><td class="num">0.6779</td><td class="num">+0.0792</td></tr>
      <tr class="rule-mid"><td>40%</td><td class="num"><strong>0.7159</strong></td><td class="num">0.6412</td><td class="num">0.6575</td><td class="num">0.6539</td><td class="num">+0.0747</td></tr>
      <tr class="rule-bottom"><td>50%</td><td class="num"><strong>0.7123</strong></td><td class="num">0.6131</td><td class="num">0.6341</td><td class="num">0.6337</td><td class="num"><strong>+0.0992</strong></td></tr>
    </tbody>
  </table>
</div>

<p>SchemaFlexTabNet-QC degrades gracefully from 0.8520 to 0.7123 AUROC at 50% missingness, maintaining a +0.0992 advantage over Native-NaN LightGBM (and +0.0782 over Native-NaN XGBoost). We emphasize two experimental cautions: (1) baseline tree performance at 0% missingness (0.7297) reflects the specific multi-cohort holdout partition and default parameters (vs. 0.8249 in pooled 5-fold CV); and (2) comparative robustness claims remain provisional until matched hyperparameter tuning under repeated seeds is completed.</p>

<div class="fig-box">
  <img src="{fig5_b64}" alt="Figure 5: Missingness Degradation">
  <div class="fig-caption"><strong>Fig. 3.</strong> Test-time missingness degradation: Schema-flexible tokenization retains robust diagnostic capacity up to 50% missing feature rates.</div>
</div>

<h3 class="subsec-title">F. Architectural Ablation and Monotonicity (Table VII)</h3>
<p>Evaluated on the held-out multi-cohort test partition, Table VII traces the contribution of each design component. Crucially, Variant 1 (inner join) restricts inputs to the 5 shared features, whereas Variant 2 expands access to all 15 union features. Benchmarking NaN-aware LightGBM on the exact same union schema yields 0.8510 AUROC (vs. 0.6280 on inner join), proving that schema retention accounts for &approx;0.22 AUROC gain, while the schema-flexible tokenizer provides the neural attention mechanism to handle arbitrary unobserved feature subsets.</p>

<div class="tab-box">
  <div class="tab-caption">Table VII: Architectural Ablation and Monotonicity Audit</div>
  <table class="ieee-tab">
    <thead>
      <tr>
        <th>Ablation Variant</th>
        <th class="num">AUROC</th>
        <th class="num">Accuracy</th>
        <th class="num">ECE</th>
        <th class="num">Mono Violations</th>
      </tr>
    </thead>
    <tbody>
      <tr class="rule-mid"><td>1. Inner-Join Baseline</td><td class="num">0.6342</td><td class="num">59.73%</td><td class="num">0.1368</td><td class="num">15.04%</td></tr>
      <tr class="rule-mid"><td>2. + Schema-Flex Tokenizer</td><td class="num">0.8560</td><td class="num">75.77%</td><td class="num">0.1672</td><td class="num">19.23%</td></tr>
      <tr class="rule-mid"><td>3. + Feature Dropout (p=0.20)</td><td class="num">0.8604</td><td class="num">77.13%</td><td class="num">0.1008</td><td class="num">4.18%</td></tr>
      <tr class="rule-mid"><td>4. + Concept Head (&lambda;<sub>1</sub>=0.10)</td><td class="num">0.8611</td><td class="num">77.82%</td><td class="num">0.1081</td><td class="num">12.97%</td></tr>
      <tr class="rule-bottom"><td>5. <strong>Full SchemaFlex (&lambda;<sub>2</sub>=1.20)</strong></td><td class="num"><strong>0.8652</strong></td><td class="num"><strong>80.20%</strong></td><td class="num"><strong>0.0572</strong></td><td class="num"><strong>0.26%</strong></td></tr>
    </tbody>
  </table>
</div>

<p>Feature dropout adds +0.004 AUROC and reduces monotonicity violations from 19.2% to 4.2%. The auxiliary concept bottleneck provides +0.0007 AUROC, confirming its role is interpretive rather than discriminative. Finally, the positive-constrained skip path and autograd penalty (&lambda;<sub>2</sub> = 1.20) suppress counterfactual violations to <strong>0.26%</strong> while post-hoc temperature scaling optimizes calibrated ECE to 0.0572.</p>

<h3 class="subsec-title">G. Mondrian Conformal Fairness Across Sex (Table VIII)</h3>
<p>Table VIII presents demographic subgroup calibration on the held-out test split. Uncalibrated predictions exhibit extreme false-positive rates (FPR &approx; 0.63). Marginal split conformal prediction introduces a severe disparity: Male FPR = 0.0375 vs. Female FPR = 0.1864 (gap = 0.1489).</p>

<div class="tab-box">
  <div class="tab-caption">Table VIII: Mondrian Group-Fair Conformal Calibration Across Sex</div>
  <table class="ieee-tab">
    <thead>
      <tr>
        <th>Method</th>
        <th>Subgroup</th>
        <th class="num">Accept. Rate</th>
        <th class="num">AUROC</th>
        <th class="num">FPR</th>
        <th class="num">FNR</th>
      </tr>
    </thead>
    <tbody>
      <tr class="rule-mid"><td rowspan="2">1. Uncalibrated (t=0.50)</td><td>Male</td><td class="num">100.0%</td><td class="num">0.8078</td><td class="num">0.6337</td><td class="num">0.1101</td></tr>
      <tr class="rule-mid"><td>Female</td><td class="num">100.0%</td><td class="num">0.7804</td><td class="num">0.6338</td><td class="num">0.1176</td></tr>
      <tr class="rule-mid"><td rowspan="2">2. Marginal Split Conformal</td><td>Male</td><td class="num">76.7%</td><td class="num">0.8181</td><td class="num">0.0375</td><td class="num">0.3951</td></tr>
      <tr class="rule-mid"><td>Female</td><td class="num">79.5%</td><td class="num">0.7982</td><td class="num">0.1864</td><td class="num">0.3636</td></tr>
      <tr class="rule-mid"><td rowspan="2">3. <strong>Mondrian Group-Fair</strong></td><td>Male</td><td class="num"><strong>79.5%</strong></td><td class="num"><strong>0.8417</strong></td><td class="num"><strong>0.1733</strong></td><td class="num"><strong>0.2500</strong></td></tr>
      <tr class="rule-bottom"><td>Female</td><td class="num"><strong>80.7%</strong></td><td class="num"><strong>0.7982</strong></td><td class="num"><strong>0.2456</strong></td><td class="num"><strong>0.2143</strong></td></tr>
    </tbody>
  </table>
</div>

<p>Mondrian calibration adjusts subgroup operating thresholds, yielding Male FPR = 0.1733 and Female FPR = 0.2456 (gap = 0.0723). This <strong>halves the false-positive disparity</strong> (0.1489 &rarr; 0.0723, a 51.4% reduction), though it does not eliminate it entirely.</p>

<div class="fig-box">
  <img src="{fig6_b64}" alt="Figure 6: Mondrian Group-Fair Calibration">
  <div class="fig-caption"><strong>Fig. 4.</strong> Mondrian group-fair calibration: Balancing subgroup operating thresholds mitigates false-positive disparity between sexes.</div>
</div>

<h3 class="subsec-title">H. Cross-Hospital Federated Learning Simulation (Table IX)</h3>
<p>Table IX benchmarks Schema-Aware Masked FedAvg over 10 communication rounds across four hospital nodes with non-overlapping schemas.</p>

<div class="tab-box">
  <div class="tab-caption">Table IX: Cross-Hospital Federated Learning Simulation</div>
  <table class="ieee-tab">
    <thead>
      <tr>
        <th>Deployment Framework</th>
        <th>Privacy Mechanism</th>
        <th class="num">AUROC</th>
        <th class="num">Gap to Central</th>
      </tr>
    </thead>
    <tbody>
      <tr class="rule-mid"><td>1. Isolated Local Hospital Models</td><td>Local Data Only</td><td class="num">0.7461</td><td class="num">-0.1285</td></tr>
      <tr class="rule-mid"><td>2. <strong>Schema-Aware FedAvg (10 Rds)</strong></td><td>Masked Token Weights</td><td class="num"><strong>0.7671</strong></td><td class="num"><strong>-0.1075</strong></td></tr>
      <tr class="rule-bottom"><td>3. Centralized Pooled Training</td><td>Full EHR Sharing</td><td class="num">0.8746</td><td class="num">0.0000</td></tr>
    </tbody>
  </table>
</div>

<p>Federation improves AUROC from 0.7461 (isolated) to 0.7671, closing <strong>16.3% of the performance gap</strong> to centralized pooling (0.8746). We note that while Schema-FedAvg shares only token weights, it provides no formal differential privacy (&epsilon;, &delta;) or cryptographic secure aggregation guarantees.</p>

<div class="fig-box">
  <img src="{fig7_b64}" alt="Figure 7: Federated Convergence">
  <div class="fig-caption"><strong>Fig. 5.</strong> Cross-hospital federated convergence: Schema-FedAvg converges stably over 10 rounds without gradient dilution.</div>
</div>

<h3 class="subsec-title">I. Clinical Decision Curve Analysis (Fig. 6)</h3>
<p>Decision Curve Analysis confirms net clinical benefit across routine threshold probabilities, outperforming uniform intervention and biopsy strategies across the entire clinically actionable range (pt = 0.05 to 0.30).</p>

<div class="fig-box">
  <img src="{fig3_b64}" alt="Figure 3: Decision Curve Analysis">
  <div class="fig-caption"><strong>Fig. 6.</strong> Clinical Decision Curve Analysis: Net benefit curves demonstrate positive utility across all realistic decision thresholds.</div>
</div>

<h2 class="sec-title">VII. Discussion and Clinical Limitations</h2>

<h3 class="subsec-title">A. Outcome Construct Divergence</h3>
<p>A central validity limitation in multi-cohort cardiovascular AI is outcome construct divergence. The merged binary label conflates three distinct clinical targets:</p>
<p class="no-indent">1) <em>Prevalent CVD at examination (Sulianova CVD, N=68,568):</em> Cross-sectional clinical history and screening findings.</p>
<p class="no-indent">2) <em>Angiographic CAD (UCI cohorts, Z-Alizadeh Sani, N=1,207):</em> Acute objective visualization of &ge; 50% luminal stenosis on catheterization.</p>
<p class="no-indent">3) <em>10-Year Incident CHD events (Framingham Study, N=4,238):</em> Prospective longitudinal event occurrence [17].</p>
<p>Predicting 10-year future events from baseline vitals is a prognostic survival task, whereas diagnosing acute luminal stenosis is an anatomical diagnostic task. This construct divergence directly explains why Framingham transfer is flat (&approx;0.678 AUROC across all models) and why conformal coverage drops to 68.8%, with accepted accuracy at 51.2%. Future multi-cohort benchmarks must avoid naively pooling disparate clinical targets or employ construct-specific prediction heads.</p>

<h3 class="subsec-title">B. Explicit Methodological Limitations</h3>
<p class="no-indent">&bull; <em>Conformal validity under cohort shift:</em> Source calibration yields coverage of 68.8%–88.4% against nominal 90%, bounded by accepted accuracy under Eq. (2); guarantees fail under distribution shift without local adaptation.</p>
<p class="no-indent">&bull; <em>Statistical power:</em> Small sample sizes (<em>N &approx; 300</em>) limit power; non-significance after Holm correction does not establish equivalence.</p>
<p class="no-indent">&bull; <em>Comparative tree robustness:</em> Baseline performance at 0% missingness reflects partition differences; multi-seed tuning is required for definitive claims.</p>
<p class="no-indent">&bull; <em>Reproduction scope:</em> Standard tabular models achieve &approx;73% on published features, but this does not constitute an architecture-faithful reproduction of unreleased neural code.</p>
<p class="no-indent">&bull; <em>Patient identity verification:</em> Feature matching demonstrates record overlap, but cannot verify biological identity without hospital records.</p>

<h2 class="sec-title">VIII. Conclusion</h2>
<p>This study proves that headline &gt;90% cardiovascular risk stratification claims on standard benchmarks are not explainable by duplicate contamination alone, but reflect reporting contradictions in published confusion matrices and an inability to reproduce such performance under standard supervised learning (73.48% empirical ceiling). Furthermore, standard inner-join merges discard vital clinical biomarkers, collapsing out-of-cohort performance.</p>
<p>To resolve these challenges, we introduced <strong>SchemaFlexTabNet-QC</strong>, a schema-flexible tabular transformer that operates across arbitrary feature sets, enforces clinical monotonicity (&lt;0.5% violations), provides temperature-scaled conformal abstention, and enables cross-hospital federated training. Rigorous evaluation against Union-Schema LightGBM demonstrates that while deep tabular transformers offer advantages under high missingness (+0.099 AUROC at 50% dropout), properly tuned tree models remain competitive on static tabular data. Open science and forensic auditing are essential to ensure clinical AI models are statistically reliable, clinically valid, and reproducible.</p>

<div class="references">
  <h2 class="sec-title" style="margin-top: 8px;">References</h2>
  <ol>
    <li>World Health Organization, "Cardiovascular diseases (CVDs) fact sheet," Tech. Rep., 2021.</li>
    <li>M. Mahmud, S. K. Das, M. A. Kashem, and M. M. Rahman, "A clinical decision support system using deep tabular learning and quality control for cardiovascular disease prediction," <em>IEEE Open J. Comput. Soc.</em>, vol. 6, pp. 112–124, 2025.</li>
    <li>S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," <em>Patterns</em>, vol. 4, no. 9, p. 100804, 2023.</li>
    <li>Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, "Revisiting deep learning models for tabular data," in <em>NeurIPS</em>, vol. 34, pp. 18932–18943, 2021.</li>
    <li>S. Ö. Arik and T. Pfister, "TabNet: Attentive interpretable tabular learning," in <em>Proc. AAAI</em>, vol. 35, no. 8, pp. 6679–6687, 2021.</li>
    <li>B. Barz and J. Denzler, "Do we train on test data? Purging CIFAR of potential leaks," in <em>Proc. IEEE/CVF WACV</em>, pp. 3010–3019, 2020.</li>
    <li>R. Detrano et al., "International application of a new probability algorithm for the diagnosis of coronary artery disease," <em>Am. J. Cardiol.</em>, vol. 64, no. 5, pp. 304–310, 1989.</li>
    <li>V. Vovk, A. Gammerman, and G. Shafer, <em>Algorithmic Learning in a Random World</em>. Springer, 2005.</li>
    <li>A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," <em>Found. Trends Mach. Learn.</em>, vol. 16, no. 4, pp. 494–591, 2023.</li>
    <li>C. Gupta, A. Podkopaev, and A. Ramdas, "Distribution-free binary classification: Prediction sets, confidence intervals and calibration," in <em>NeurIPS</em>, vol. 34, pp. 3711–3723, 2021.</li>
    <li>R. J. Tibshirani, R. F. Barber, E. Candes, and A. Ramdas, "Conformal prediction under covariate shift," in <em>NeurIPS</em>, vol. 32, 2019.</li>
    <li>R. F. Barber, E. J. Candes, A. Ramdas, and R. J. Tibshirani, "Predictive inference with the jackknife+," <em>Ann. Statist.</em>, vol. 49, no. 1, pp. 486–507, 2021.</li>
    <li>E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, "Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach," <em>Biometrics</em>, vol. 44, no. 3, pp. 837–845, 1988.</li>
    <li>B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," in <em>AISTATS</em>, pp. 1273–1282, 2017.</li>
    <li>G. S. Collins et al., "Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: TRIPOD+AI statement," <em>BMJ</em>, vol. 385, p. e078378, 2024.</li>
    <li>R. Alizadehsani et al., "A database of coronary artery disease patients based on angiography," <em>Comput. Methods Programs Biomed.</em>, vol. 111, no. 1, pp. 52–61, 2013.</li>
    <li>T. R. Dawber, G. F. Meadors, and F. E. Moore, "Epidemiological approaches to heart disease: the Framingham Study," <em>Am. J. Public Health</em>, vol. 41, no. 3, pp. 279–281, 1951.</li>
    <li>Z. Janosi, W. Steinbrunn, K. Pfisterer, and R. Detrano, "Hungarian Institute of Cardiology dataset," UCI Machine Learning Repository, 1988.</li>
    <li>M. Pfisterer, W. Steinbrunn, and R. Detrano, "Heart disease dataset: Switzerland and Long Beach VA medical centers," UCI Machine Learning Repository, 1988.</li>
    <li>G. Somepalli, M. Goldblum, A. Bollapragada, C. B. Bruss, and T. Goldstein, "SAINT: Improved neural networks for tabular data via row and column attention," <em>arXiv:2106.01342</em>, 2021.</li>
    <li>C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in <em>Proc. ICML</em>, pp. 1321–1330, 2017.</li>
    <li>A. J. Vickers and E. B. Elkin, "Decision curve analysis: A novel method for evaluating prediction models," <em>Med. Decis. Making</em>, vol. 26, no. 6, pp. 565–574, 2006.</li>
  </ol>
</div>

</div>

</body>
</html>
"""

with open(HTML_OUT, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"[OK] Generated comprehensive HTML manuscript at: {HTML_OUT}")

cmd = [
    CHROME_PATH,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={PDF_OUT}",
    str(HTML_OUT),
]

print(f"[INFO] Invoking Chrome Headless to compile PDF...")
proc = subprocess.run(cmd, capture_output=True, text=True)
if proc.returncode == 0 and PDF_OUT.exists() and PDF_OUT.stat().st_size > 0:
    print(f"[SUCCESS] Publication PDF compiled successfully!")
    print(f"  -> File: {PDF_OUT}")
    print(f"  -> Size: {PDF_OUT.stat().st_size / 1024:.1f} KB")
    reader = pypdf.PdfReader(str(PDF_OUT))
    print(f"  -> Total Page Count: {len(reader.pages)} pages")
    import shutil
    target_copy = BASE_DIR / "DS_Research_our_paper.pdf"
    shutil.copy2(PDF_OUT, target_copy)
    print(f"  -> Synchronized copy to: {target_copy}")
else:
    print(f"[ERROR] Chrome PDF compilation failed: {proc.stderr}")
