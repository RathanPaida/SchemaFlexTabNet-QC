# Forensic Benchmarking and Schema-Flexible Clinical Tabular Learning: Resolving Data Leakage, Duplicate Contamination, and Inner-Join Collapse in Multi-Cohort Cardiovascular Risk Stratification

**Author:** Raja Rathan Paida  
**Affiliation:** Department of Artificial Intelligence, Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat 395007, Gujarat, India  
**Correspondence:** `paida.rajarthan@gmail.com`  
**Open-Source Repository:** [https://github.com/RathanPaida/SchemaFlexTabNet-QC](https://github.com/RathanPaida/SchemaFlexTabNet-QC)  
**Target Venues:** IEEE Journal of Biomedical and Health Informatics (IEEE JBHI) / IEEE Transactions on Biomedical Engineering / Computers in Biology and Medicine  
**Status:** Complete Empirical Manuscript with Full Reproducibility Suite  

---

## Abstract

Recent high-profile literature in machine learning for cardiovascular disease (CVD) risk stratification reports diagnostic accuracies exceeding $90\%$ on public benchmarks (e.g., $90.1\%$ in Mahmud et al., 2025). In this paper, we conduct a forensic audit of these claims, uncovering critical methodological vulnerabilities in current clinical tabular workflows:
1. **Duplicate Contamination and Leakage:** The widely used Kaggle Heart Disease dataset contains $70.5\%$ exact duplicate records ($723$ of $1,025$ rows). Standard random $80/20$ train-test splits leak $97.1\%$ of test patients into training, yielding an artificial $98.54\%$ accuracy that plummets to $77.05\%$ upon patient-level deduplication. However, we demonstrate mathematically and empirically that duplicate leakage alone cannot explain the $90.1\%$ claim across the combined $71,025$-patient composite (where duplicates constitute only $\sim 1.02\%$ of records); an exhaustive audit reveals internal discrepancies in published confusion matrices (implying $91.35\%$ vs. $89.5\%$ accuracy) and class balance contradictions. Standard supervised models trained on this $71,025$-patient composite achieve an empirical ceiling of $73.48\%$ accuracy ($0.7978$ AUROC).
2. **The Inner-Join Collapse:** Naive multi-center merges discard non-overlapping clinical biomarkers (e.g., ST depression, peak heart rate, fluoroscopy), causing out-of-cohort zero-shot AUROC to collapse to $0.5353$–$0.5917$.

To solve these challenges, we introduce **SchemaFlexTabNet-QC**, a tabular transformer framework featuring:
- **Hemodynamic Plausibility Gatekeeper:** Filters uncalibrated cuff artifacts ($\text{SBP} \in [60, 260]$, $\text{DBP} \in [40, 160]$ mmHg, $\text{PP} \ge 10$ mmHg).
- **Schema-Flexible Feature Tokenization:** Maps each clinical observation to an independent identity-value token ($T_{i,j} = \mathbf{e}_{\text{name}}(f_j) + \phi_{\text{val}}(x_{i,j})$) with masked self-attention and random feature dropout ($p_{\text{mask}}=0.20$), accommodating arbitrary feature subsets without column pruning or imputation.
- **Positive-Constrained Monotonic Residual Skip Path:** Combines an explicit non-negative linear skip path with autograd gradient penalties ($\lambda_{\text{mono}} = 1.20$), reducing clinical monotonicity violations on harmful biomarkers from $15.04\%$ to $0.26\%$.
- **Conformal Abstention & Mondrian Fairness:** Deploys Temperature-Scaled Split Conformal Abstention alongside Mondrian group-fair calibration. We analyze conformal coverage under cross-cohort distribution shift, demonstrating that exchangeability is violated and showing that Mondrian calibration halves sex false-positive disparities ($0.148 \to 0.072$).
- **Decentralized Schema-FedAvg:** Implements Schema-Aware Masked Federated Averaging across non-overlapping hospital schemas, closing $16.3\%$ of the gap to centralized pooled training without transmitting raw patient records.

Evaluated on $74,013$ deduplicated patients across six independent cohorts under strict Leave-One-Cohort-Out (LOCO) testing, SchemaFlexTabNet-QC matches or outperforms Union-Schema NaN-aware LightGBM (mean gain $+0.0229$ AUROC; $0.8641$ on Hungarian and $0.7043$ on Z-Alizadeh Sani). Under conformal abstention ($\alpha=0.10$), zero-shot AUROC rises to **$0.8926$** ($85.15\%$ accuracy, $78.2\%$ acceptance rate) on Hungarian and **$0.8755$** on Cleveland. Under $50\%$ test-time feature dropout, SchemaFlexTabNet-QC maintains $0.7123$ AUROC, retaining an advantage over Native-NaN LightGBM ($0.6131$) and median-imputed baselines ($0.6337$). Finally, we discuss construct divergence across prevalent CVD, angiographic disease, and incident 10-year coronary events, establishing rigorous reporting practices for clinical tabular AI.

---

## 1. Introduction

Cardiovascular diseases (CVDs) remain the principal etiology of mortality worldwide, claiming an estimated 17.9 million lives each year [1]. While machine learning holds transformative promise for early risk detection, translating algorithms from single-institution research prototypes to multi-center clinical deployments faces persistent bottlenecks: distribution shift, missing clinical panels, unharmonized electronic health record (EHR) schemas, and evaluation artifacts [15].

In recent years, several published investigations have reported diagnostic accuracies exceeding $90.1\%$ for cardiovascular prediction using deep tabular architectures (notably Mahmud et al., 2025 [2] and related works). However, clinical epidemiologists and translational informatics researchers frequently treat $>90\%$ claims on routine clinical vitals with skepticism [3]. On massive, real-world population screening cohorts (e.g., the 70,000-patient Sulianova dataset [2]), the empirical Bayes-optimal performance ceiling established by modern tree ensembles (LightGBM, XGBoost, CatBoost) and deep neural architectures (TabNet, FT-Transformer) hovers consistently between $73.4\%$ and $76.0\%$ accuracy ($0.818$–$0.825$ AUROC) [4, 5].

When such performance discrepancies emerge, forensic auditing is essential to determine whether the reported accuracy reflects genuine algorithmic breakthroughs or hidden methodological pitfalls [6]. In this paper, we conduct a comprehensive forensic investigation into recent cardiovascular tabular benchmarks and identify two structural failure modes:
1. **Duplicate Contamination and Leakage Fallacy:** The popular Kaggle Heart Disease dataset (uploaded by johnsmith88) consists of 1,025 records. However, 723 rows ($70.5\%$) are identical duplicates of other rows. When evaluated with standard random 80/20 train-test splits, $97.1\%$ of test patients are present in the training split, inflating accuracy to an artificial $98.54\%$ ($0.9980$ AUROC). Upon rigorous patient deduplication, accuracy falls to $77.05\%$ ($0.8593$ AUROC). Yet, as we demonstrate in Section 2, duplicate leakage alone *cannot mathematically explain* the 90.1% accuracy claimed across the combined 71,025-patient composite (1,025 Kaggle + 70,000 Sulianova rows), because the 723 duplicates represent only $\sim 1.02\%$ of total records. An audit of published manuscripts reveals significant internal reporting contradictions in confusion matrices and class distributions, while rigorous 70/15/15 supervised reproductions achieve an empirical ceiling of $73.48\%$ accuracy.
2. **The Destructive Inner-Join Merge:** When integrating tertiary cardiology cohorts (e.g., UCI Cleveland, Hungarian, Long Beach VA, Switzerland, and Z-Alizadeh Sani) with large-scale screening databases (Sulianova CVD), conventional clinical data engineering performs an inner-join merge on shared columns. This operation discards diagnostic biomarkers that are not universally collected across all hospitals—specifically ST depression (`oldpeak`), peak exercise heart rate (`thalach`), and fluoroscopy vessels (`ca`). Discarding these predictive features collapses out-of-cohort zero-shot AUROC to $0.5353$–$0.5917$.

### Key Methodological Contributions
1. **Mathematical and Forensic Audit of the 90.1% Claim:** We prove that duplicate contamination leaks $97.1\%$ of test samples in isolated 1,025-row datasets, but demonstrate that across the 71,025-row composite, duplicates represent only $\sim 1.02\%$ of data, which mathematically cannot shift accuracy by $+15\%$. We document internal inconsistencies in published figures (confusion matrix implying $91.35\%$ vs. text implying $89.5\%$) and report empirical 70/15/15 reproduction benchmarks ($73.48\%$ accuracy).
2. **Schema-Flexible Tokenization:** We formulate a tabular transformer architecture that tokenizes patient records into dynamic feature sets, eliminating column-pruning inner joins and preserving $100\%$ of hospital-specific diagnostic features.
3. **Physiological Guarantees via Monotonic Skip Paths:** We construct an explicit positive-constrained monotonic skip path combined with autograd gradient penalties ($\lambda_{\text{mono}} = 1.20$), slashing counterfactual monotonicity violations on harmful biomarkers from $15.04\%$ to $0.26\%$.
4. **Conformal Safety and Cohort Shift Analysis:** We evaluate split conformal prediction and Mondrian fairness under Leave-One-Cohort-Out (LOCO) testing. We document that cross-cohort distribution shift violates exchangeability, report empirical set coverage alongside acceptance rates, and demonstrate that Mondrian calibration halves sex false-positive disparities ($0.148 \to 0.072$).
5. **Fair DeLong Significance vs. Union-Schema LightGBM:** We contrast SchemaFlexTabNet-QC against the proper, union-schema NaN-aware LightGBM baseline with Holm-Bonferroni correction, providing an objective assessment of deep tabular transformers versus gradient-boosted trees.

---

## 2. Forensic Audit of Baseline Literature

### 2.1 Kaggle Duplicate Contamination (johnsmith88)
The public Kaggle Heart Disease dataset (`heart.csv`, uploaded by user johnsmith88) contains $N = 1,025$ patient records across 14 clinical attributes. To investigate the internal composition of this dataset, we executed an exact-match hash audit across all clinical features (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`).

The audit revealed that only 302 unique patient signatures exist within the file. The remaining 723 rows ($70.54\%$) are exact duplicates resulting from non-random sampling with replacement of the original 1989 UCI Cleveland Clinic cohort [7]. When researchers employ standard random 80/20 train/test splits, test records are overwhelmingly drawn from the duplicate pool:
$$P(\mathbf{x}_{\text{test}} \in \mathcal{D}_{\text{train}}) = \frac{199}{205} = 97.07\%$$
As shown in Table 1, an unregularized tree ensemble or neural network evaluated on this split memorizes patient identities, yielding an artificial $98.54\%$ accuracy ($0.9980$ AUROC). Pruning duplicates and evaluating across unique patients collapses accuracy to $77.05\%$ ($0.8593$ AUROC).

### 2.2 The 90.1% Claim on the 71,025-Patient Composite
In Mahmud et al. (2025) [2], the authors claim a diagnostic accuracy of $90.1\%$ (AUROC $0.963$) on a combined cardiovascular dataset. The combined dataset was constructed by pooling the 1,025-row Kaggle Heart dataset with the 70,000-row Sulianova Cardiovascular dataset [2], totaling $N = 71,025$ rows. The authors specify a 70/15/15 train/validation/test split, yielding a test set of exactly:
$$N_{\text{test}} = 0.15 \times 71,025 = 10,653.75 \approx 10,653 \text{ rows}$$
The exact match of 10,653 test rows confirms that the 1,025-row Kaggle file was pooled into the dataset. However, attributing the $90.1\%$ accuracy solely to memorization of the 723 duplicate rows is mathematically flawed:
$$\text{Duplicate Fraction} = \frac{723}{71,025} = 1.018\% \approx 1.02\%$$
Even under the extreme assumption that every duplicate row is classified with $100\%$ accuracy, a $1.02\%$ contaminated subset can alter the global test accuracy by at most:
$$\Delta \text{Acc}_{\text{max}} \le 0.0102 \times (1.00 - 0.73) \approx +0.28\%$$
The remaining $98.98\%$ of the dataset consists of Sulianova records. On this dataset, extensive multi-model benchmarks (Table 2) demonstrate that state-of-the-art models (LightGBM, XGBoost, CatBoost, TabNet) saturate between $73.4\%$ and $76.0\%$ accuracy.

To understand why $90.1\%$ was reported, we conducted a forensic analysis of the published manuscript [2]:
- **Confusion Matrix Discrepancy:** In Fig. 2 of Mahmud et al., the confusion matrix displays: True Positives ($\text{TP}$) = 5,044, True Negatives ($\text{TN}$) = 4,688, False Positives ($\text{FP}$) = 444, and False Negatives ($\text{FN}$) = 477. The sum of these cells is $5,044 + 4,688 + 444 + 477 = 10,653$ (matching the 15% test set). However, computing accuracy directly from this matrix yields:
  $$\text{Accuracy} = \frac{5,044 + 4,688}{10,653} = \frac{9,732}{10,653} = 91.35\%$$
  contradicting the $90.1\%$ stated in the abstract and text.
- **Textual Inconsistency:** The narrative in Section 4.2 of Mahmud et al. describes a different set of classification counts that yields approximately $89.5\%$ accuracy.
- **Class Distribution Discrepancy:** In the confusion matrix, disease prevalence is $51.8\%$, whereas the authors' own data preprocessing section specifies a prevalence filter altering the balance.

| Experimental Setup | Total $N$ | Test $N$ | Accuracy (%) | AUROC | Leakage Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Kaggle Leaked Split (80/20) | 1,025 | 205 | 98.54% | 0.9980 | 97.1% |
| Kaggle Deduplicated Split | 302 | 61 | 77.05% | 0.8593 | 0.0% |
| **Mahmud Reported Baseline** | **71,025** | **10,653** | **90.10%**\* | **0.9630** | — |
| Combined Leaked (71,025) Reproduction | 69,692 | 10,454 | 73.48% | 0.7978 | 1.02% |
| Combined Deduplicated Reproduction | 68,969 | 10,346 | 73.18% | 0.7954 | 0.0% |
| Sulianova Cohort Alone | 68,667 | 10,301 | 73.42% | 0.7987 | 0.0% |

*\*Fig. 2 confusion matrix in Mahmud et al. implies 91.35% accuracy; text reports 90.10% and 89.50%.*

To empirically test whether standard supervised learning on this combined schema can produce $>90\%$ accuracy, we implemented the 70/15/15 split on the exact combined 71,025-row dataset, the deduplicated dataset, and Sulianova alone under identical hemodynamic filtering (Table 1). The combined leaked pipeline yields **$73.48\%$ accuracy** ($0.7978$ AUROC), the deduplicated pipeline yields **$73.18\%$**, and Sulianova alone yields **$73.42\%$**. The inclusion of duplicates shifts accuracy by only $+0.30\%$. We conclude that while duplicate contamination in the Kaggle file is real, the $90.1\%$ claim cannot be reproduced under standard supervised training on these clinical features, reflecting undocumented preprocessing steps or reporting discrepancies in the original publication.

### 2.3 Cross-Cohort Contamination: UCI Cleveland vs. Statlog
A secondary forensic pitfall in the literature involves evaluating Leave-One-Cohort-Out transfer on UCI Statlog ($N=270$) after training on UCI Cleveland ($N=303$). A biometric signature audit across six core physiological parameters revealed that all 270 Statlog records are identical clones of Cleveland Clinic patients. Testing on Statlog while training on Cleveland constitutes cross-cohort patient leakage [3]. In our benchmark, all 270 duplicate Statlog rows are permanently purged.

### 2.4 The Destructive Inner-Join Merge
In multi-center healthcare systems, different clinical sites collect distinct lab panels. When merging datasets, standard pipelines perform an inner join, discarding unshared columns. In cardiovascular care, screening datasets (e.g., Sulianova) collect non-invasive vitals (blood pressure, cholesterol, glucose, BMI), while tertiary cardiac catheterization labs (e.g., UCI cohorts, Z-Alizadeh Sani) measure coronary fluoroscopy (`ca`), exercise ST depression (`oldpeak`), and peak heart rate (`thalach`).

As detailed in Table 3, performing an inner join forces tertiary models to rely solely on non-invasive vitals. When evaluated zero-shot on held-out hospitals, inner-join LightGBM collapses to $0.5917$ AUROC on Hungarian and $0.5353$ on VA + Switzerland. Discarding specialized diagnostic columns severely impairs zero-shot clinical utility.

---

## 3. Proposed Architecture: SchemaFlexTabNet-QC

### 3.1 Hemodynamic Plausibility Gatekeeper
EHR databases contain uncalibrated cuff artifacts and sensor transcription errors. Prior to tokenization, input vitals pass through a physiological gatekeeper based on human hemodynamic limits:
$$\text{SBP} \in [60, 260]\text{ mmHg}, \quad \text{DBP} \in [40, 160]\text{ mmHg}$$
$$\text{PP} = \text{SBP} - \text{DBP} \ge 10\text{ mmHg}$$
Records violating these physical constraints (e.g., inverted $\text{SBP} < \text{DBP}$) are filtered. In Sulianova, this identifies 1,374 physiologically corrupt rows.

### 3.2 Schema-Flexible Feature Tokenization
Let a patient $i$ be characterized by an arbitrary observed feature subset $\mathcal{F}_i \subseteq \{f_1, f_2, \dots, f_M\}$ drawn from the global union of clinical attributes ($M=22$). Rather than arranging inputs into a static matrix with imputed zeros, each observed feature $j \in \mathcal{F}_i$ is transformed into a continuous vector token $T_{i,j} \in \mathbb{R}^d$:
$$T_{i,j} = \mathbf{e}_{\text{name}}(f_j) + \phi_{\text{val}}(x_{i,j})$$
where $\mathbf{e}_{\text{name}}(f_j) \in \mathbb{R}^d$ is a learned feature-identity embedding, and $\phi_{\text{val}}(x_{i,j})$ is a feature-specific linear projection:
$$\phi_{\text{val}}(x_{i,j}) = x_{i,j} \cdot \mathbf{w}_j + \mathbf{b}_j, \quad \mathbf{w}_j, \mathbf{b}_j \in \mathbb{R}^d$$
For numerical stability, continuous variables are z-score standardized using training statistics. Prepending a learned $[\text{CLS}] \in \mathbb{R}^d$ token yields a variable-length sequence:
$$\mathbf{Z}_0 = [ [\text{CLS}], T_{i,j_1}, T_{i,j_2}, \dots, T_{i,j_{|\mathcal{F}_i|}} ] \in \mathbb{R}^{(1 + |\mathcal{F}_i|) \times d}$$
The sequence is processed by an $L$-layer Transformer Encoder ($L=2$, $H=4$ attention heads, $d=64$, $d_{\text{ff}}=128$, dropout $0.10$):
$$\mathbf{Z}'_l = \text{LayerNorm}(\mathbf{Z}_{l-1} + \text{MHA}(\mathbf{Z}_{l-1}, \mathbf{M}_i))$$
$$\mathbf{Z}_l = \text{LayerNorm}(\mathbf{Z}'_l + \text{FFN}(\mathbf{Z}'_l))$$
where $\mathbf{M}_i$ is a boolean key padding mask ensuring unobserved tokens receive zero attention. To instill missing-schema robustness, we inject random feature dropout during training with probability $p_{\text{mask}} = 0.20$.

### 3.3 Positive-Constrained Monotonic Skip Path
Cardiovascular clinicians require that established harmful biomarkers (e.g., advancing age, malignant systolic blood pressure, high cholesterol, severe ST depression) must not lower predicted disease risk when elevated counterfactually [15]. Because unconstrained multi-head attention can produce non-monotonic artifacts, we introduce a dual inductive bias:
1. **Positive-Constrained Skip Path:** We define a monotonic linear skip branch for the subset of established harmful biomarkers $\mathcal{M} = \{\text{age}, \text{ap\_hi}, \text{trestbps}, \text{chol}, \text{oldpeak}\}$:
   $$f_{\text{mono}}(\mathbf{x}_i) = \sum_{k \in \mathcal{M}} \text{softplus}(w_k) \cdot x_{i,k} \cdot \mathbb{I}(x_{i,k} \text{ observed})$$
   where $\text{softplus}(w) = \ln(1 + e^w) > 0$ strictly enforces non-negative coefficients.
2. **Autograd Monotonicity Regularization:** We compute the exact input gradient via automatic differentiation and penalize negative directional gradients:
   $$\mathcal{L}_{\text{mono}} = \frac{1}{|\mathcal{M}|} \sum_{k \in \mathcal{M}} \text{ReLU}\left( -\frac{\partial \hat{y}_i}{\partial x_{i,k}} \right) \cdot \mathbb{I}(x_{i,k} \text{ observed})$$

The final logit is the sum of the transformer $[\text{CLS}]$ projection and the monotonic branch:
$$\hat{y}_i = \sigma\left( \mathbf{W}_{\text{out}} \mathbf{z}_{\text{CLS}} + b_0 + f_{\text{mono}}(\mathbf{x}_i) \right)$$
The total loss function is:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{y}_i, y_i) + \lambda_1 \mathcal{L}_{\text{concept}} + \lambda_2 \mathcal{L}_{\text{mono}}$$
where $\mathcal{L}_{\text{concept}}$ is an auxiliary concept bottleneck predicting Stage 2 Hypertension ($\text{SBP} \ge 140$ or $\text{DBP} \ge 90$ mmHg) with $\lambda_1 = 0.10$, and $\lambda_2 = 1.20$.

---

## 4. Conformal Prediction and Fairness Under Cohort Shift

### 4.1 Conformal Abstention and Exchangeability Violations
In Split Conformal Prediction [8, 9], non-conformity scores $s_i = 1 - \hat{P}(Y_i = y_i \mid \mathbf{X}_i)$ are computed on a calibration split $\mathcal{D}_{\text{cal}}$. Given an error tolerance $\alpha \in (0, 1)$, the conformal quantile is:
$$\hat{q} = \text{Quantile}\left( \frac{\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil}{n_{\text{cal}}}; \{s_i\}_{i=1}^{n_{\text{cal}}} \right)$$
For a new patient $\mathbf{X}_{n+1}$, the prediction set is constructed as:
$$\mathcal{C}(\mathbf{X}_{n+1}) = \left\{ y \in \{0, 1\} \mid 1 - \hat{P}(Y = y \mid \mathbf{X}_{n+1}) \le \hat{q} \right\}$$

When calibration and testing occur on independent, identically distributed samples from the same cohort, exchangeability guarantees marginal set coverage:
$$\mathbb{P}(Y_{n+1} \in \mathcal{C}(\mathbf{X}_{n+1})) \ge 1 - \alpha$$
However, in Leave-One-Cohort-Out clinical deployments, calibration is performed on source cohorts while testing occurs on a distinct held-out hospital. Under this cross-cohort shift:
$$P_{\text{source}}(\mathbf{X}, Y) \neq P_{\text{target}}(\mathbf{X}, Y)$$
Exchangeability is mathematically violated [11, 12]. As demonstrated in Table 5, empirical target set coverage drops below the nominal $1-\alpha = 0.90$ level (ranging from $84.2\%$ on Z-Alizadeh Sani to $88.7\%$ on Hungarian). Calibrating on a small held-out calibration partition from the target cohort restores exact coverage ($\ge 90.1\%$).

In our clinical reporting, we strictly distinguish between:
- **Conformal Set Coverage:** The probability that the true label belongs to the prediction set, $\mathbb{P}(Y \in \mathcal{C}(X))$.
- **Acceptance Rate (Selection Rate):** The fraction of decisive singleton predictions, $|\mathcal{C}(X)| = 1$, where the model does not abstain.

### 4.2 Mondrian Group-Fair Conformal Calibration
To mitigate disparities across demographic groups $G \in \{\text{Male}, \text{Female}\}$, standard global thresholding can yield unequal false-positive rates. Mondrian conformal calibration computes group-conditional quantiles $\hat{q}_g$ [10]:
$$\mathbb{P}(Y_{n+1} \in \mathcal{C}(\mathbf{X}_{n+1}) \mid G_{n+1} = g) \ge 1 - \alpha, \quad \forall g \in \{M, F\}$$
By calibrating subgroup decision thresholds against equalized operating targets on the calibration split, Mondrian calibration substantially mitigates disparities across sexes.

### 4.3 Schema-Aware Federated Learning (Schema-FedAvg)
In multi-hospital consortia, patient records cannot leave local firewalls. When hospitals possess non-identical schemas, naive FedAvg [14] fails because clients update disjoint weight parameters. We implement **Schema-Aware Masked FedAvg**: each hospital $k$ trains locally on its observed token subset $\mathcal{F}^{(k)}$. During central aggregation, global parameter tensors are updated using presence-masked weighting:
$$\mathbf{W}_{t+1}[j] = \frac{\sum_{k \in \mathcal{S}_j} n_k \mathbf{W}_{t,k}[j]}{\sum_{k \in \mathcal{S}_j} n_k}$$
where $\mathcal{S}_j = \{k \mid \text{feature } j \in \mathcal{F}^{(k)}\}$. Hospitals collaborate without transmitting raw patient rows or imputing missing columns.

---

## 5. Experimental Setup & Benchmarks

### 5.1 Multi-Cohort Harmonization
We assemble and harmonize six independent cardiovascular cohorts spanning 74,013 clean, non-overlapping patients:
1. **Sulianova Population Screening Cohort** ($N = 68,568$ clean): Primary screening cohort in Russia recording demographics, blood pressure, cholesterol, glucose, and lifestyle vitals. Disease endpoint: prevalent cardiovascular disease survey.
2. **UCI Cleveland Clinic** ($N = 303$ clean): Tertiary cardiac catheterization cohort. Disease endpoint: angiographic coronary artery disease ($\ge 50\%$ diameter stenosis) [7].
3. **UCI Hungarian Institute of Cardiology** ($N = 293$): Angiographic cohort in Budapest [18].
4. **UCI Long Beach VA & Switzerland** ($N = 308$ pooled): High-risk veterans and Swiss cardiology patients [19].
5. **Z-Alizadeh Sani Cohort** ($N = 303$): Tertiary catheterization registry from Tehran Heart Center [16].
6. **Framingham Heart Study** ($N = 4,238$): Longitudinal epidemiological cohort [17]. Disease endpoint: 10-year incident coronary heart disease event.

### 5.2 Baseline Multi-Model Benchmarking (Table 2)
We benchmark seven competitive tabular architectures across Stratified 5-Fold Cross-Validation on the harmonized multi-cohort dataset ($N=74,013$, Statlog duplicates excluded). Configuration A evaluates raw features (10 features); Configuration B includes engineered physiological markers ($\text{BMI}$, pulse pressure, ATP-III tiers).

| Model Architecture | Raw AUROC | Raw Accuracy (%) | Meta AUROC | Meta Accuracy (%) | $\Delta$ AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.7554 | 69.16% | 0.7678 | 70.19% | +0.0124 |
| Random Forest | 0.8195 | 75.03% | 0.8244 | 75.85% | +0.0049 |
| XGBoost | 0.8186 | 75.27% | 0.8247 | 75.95% | +0.0061 |
| **LightGBM** | **0.8187** | **75.39%** | **0.8249** | **75.98%** | **+0.0062** |
| CatBoost | 0.8152 | 75.13% | 0.8215 | 75.60% | +0.0063 |
| MLP Neural Net | 0.8096 | 74.34% | 0.8110 | 74.86% | +0.0014 |
| TabNet | 0.8078 | 74.53% | 0.8105 | 74.87% | +0.0027 |

As presented in Table 2, all top-performing models saturate within a narrow performance band: LightGBM achieves $75.39\%$ accuracy ($0.8187$ AUROC) on raw features and $75.98\%$ accuracy ($0.8249$ AUROC) with meta-features. Random Forest ($75.85\%$), XGBoost ($75.95\%$), and CatBoost ($75.60\%$) demonstrate near-identical metrics. Deep tabular baselines (TabNet, MLP) achieve $74.87\%$ accuracy ($0.8105$ AUROC). This confirms that on noisy population screening data, the empirical accuracy ceiling resides at approximately $75.4\%$–$76.0\%$.

---

## 6. Comprehensive Results & Empirical Analysis

### 6.1 Multi-Cohort Leave-One-Cohort-Out Matrix (Table 3)
Table 3 presents the primary out-of-cohort generalization benchmark across five held-out evaluation targets. In each experiment, models are trained on all other cohorts and evaluated zero-shot on the target cohort.

| Held-Out Target Cohort | Model Architecture | Acceptance Rate (%) | Zero-Shot AUROC | Accuracy (%) | Brier Score | Calibrated ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **UCI Hungarian** ($N=293$) | Inner-Join LightGBM | 100.0% | 0.5917 | 55.63% | 0.2808 | 0.2160 |
| | Union-Schema LightGBM | 100.0% | 0.8423 | 74.06% | 0.1743 | 0.1325 |
| | HybridTabNet-QC Baseline | 100.0% | 0.6111 | 59.04% | 0.2581 | 0.1254 |
| | **SchemaFlexTabNet-QC (Full)** | 100.0% | **0.8641** | **76.79%** | **0.1545** | **0.1026** |
| | **SchemaFlex (Conformal $\alpha=0.10$)** | **78.2%** | **0.8926** | **85.15%** | **0.1236** | **0.0756** |
| **UCI VA + Switzerland** ($N=308$) | Inner-Join LightGBM | 100.0% | 0.5353 | 53.57% | 0.2863 | 0.3047 |
| | Union-Schema LightGBM | 100.0% | **0.7280** | 72.40% | 0.1934 | 0.2134 |
| | HybridTabNet-QC Baseline | 100.0% | 0.5451 | 70.13% | 0.2064 | 0.1921 |
| | **SchemaFlexTabNet-QC (Full)** | 100.0% | 0.7241 | **74.03%** | **0.1721** | **0.1382** |
| | **SchemaFlex (Conformal $\alpha=0.10$)** | **72.7%** | **0.7705** | **83.48%** | **0.1335** | **0.0623** |
| **UCI Cleveland (Dedup)** ($N=303$) | Inner-Join LightGBM | 100.0% | 0.5705 | 56.44% | 0.2748 | 0.1759 |
| | Union-Schema LightGBM | 100.0% | **0.8566** | **76.90%** | **0.1560** | **0.0559** |
| | HybridTabNet-QC Baseline | 100.0% | 0.6785 | 62.71% | 0.2316 | 0.0890 |
| | **SchemaFlexTabNet-QC (Full)** | 100.0% | 0.8433 | 73.60% | 0.1829 | 0.1209 |
| | **SchemaFlex (Conformal $\alpha=0.10$)** | **78.9%** | **0.8755** | **78.66%** | **0.1615** | **0.0946** |
| **Z-Alizadeh Sani** ($N=303$) | Union-Schema LightGBM | 100.0% | 0.6028 | 68.98% | 0.2131 | 0.1148 |
| | **SchemaFlexTabNet-QC (Full)** | 100.0% | **0.7043** | 62.71% | 0.2333 | 0.2072 |
| | **SchemaFlex (Conformal $\alpha=0.10$)** | **36.6%** | **0.7895** | **72.07%** | **0.2136** | **0.1921** |
| **Framingham Study** ($N=4,238$) | Union-Schema LightGBM | 100.0% | 0.6695 | **53.26%** | **0.2965** | **0.3614** |
| | **SchemaFlexTabNet-QC (Full)** | 100.0% | **0.6778** | 51.32% | 0.3213 | 0.3958 |
| | **SchemaFlex (Conformal $\alpha=0.10$)** | **64.1%** | **0.7062** | 51.23% | 0.3576 | 0.4109 |

On UCI Hungarian, SchemaFlexTabNet-QC achieves $0.8641$ AUROC ($76.79\%$ accuracy), significantly surpassing Inner-Join LightGBM ($0.5917$) and outperforming Union-Schema LightGBM ($0.8423$). Under Conformal Abstention ($\alpha=0.10$), the acceptance rate is $78.2\%$, and AUROC on the accepted cohort rises to **$0.8926$** ($85.15\%$ accuracy).

On UCI Cleveland, Union-Schema LightGBM achieves $0.8566$ AUROC vs. SchemaFlexTabNet-QC's $0.8433$ (delta $-0.0133$), demonstrating that gradient-boosted trees remain formidable baselines when trained on union schemas. However, conformal abstention on SchemaFlexTabNet-QC lifts AUROC to **$0.8755$** ($78.66\%$ accuracy).

On Z-Alizadeh Sani, SchemaFlexTabNet-QC markedly outperforms Union-Schema LightGBM ($0.7043$ vs. $0.6028$, gain $+0.1015$).

### 6.2 Rigorous DeLong Significance Testing vs. Union LightGBM (Table 4)
Previous reports compared deep models against an inner-join LightGBM baseline, reporting extreme significance ($p \approx 10^{-14}$). To eliminate straw-man comparisons, Table 4 evaluates paired DeLong statistical significance tests [13] against the true competitor: **Union-Schema NaN-Aware LightGBM**, with Holm-Bonferroni step-down correction across all five cohorts.

| Held-Out Cohort | SchemaFlex AUROC | Union LGB AUROC | $\Delta$ vs. Union | Raw $p$ (Union) | Holm $p$ (Union) | Signif. | DeLong $p$ (Hybrid) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UCI Hungarian | 0.8641 | 0.8423 | +0.0218 | 0.1578 | 0.6310 | n.s. | $< 10^{-10}$ (***) |
| UCI VA + Switz. | 0.7241 | 0.7280 | -0.0039 | 0.7984 | 1.0000 | n.s. | $5.6 \times 10^{-4}$ (**) |
| UCI Cleveland | 0.8433 | 0.8566 | -0.0133 | 0.4472 | 1.0000 | n.s. | $2.1 \times 10^{-6}$ (***) |
| Z-Alizadeh Sani | 0.7043 | 0.6028 | +0.1015 | 0.0369 | 0.1846 | n.s. | 0.8384 (n.s.) |
| Framingham | 0.6778 | 0.6695 | +0.0083 | 0.4700 | 1.0000 | n.s. | 0.7058 (n.s.) |

Against the inner-join HybridTabNet-QC, SchemaFlexTabNet-QC is overwhelmingly superior ($p < 10^{-5}$ to $p < 10^{-10}$). However, against Union-Schema LightGBM, the mean AUROC gain across the five cohorts is modest ($+0.0229$). After Holm-Bonferroni correction, the individual cohort differences do not reach $p < 0.05$. This finding reinforces the necessity of comparing against properly engineered tabular tree baselines.

### 6.3 Conformal Cohort Shift and Exchangeability Audit (Table 5)
Table 5 investigates the impact of distribution shift on conformal guarantees. When calibration is performed on source cohorts, empirical set coverage $\mathbb{P}(Y \in \mathcal{C}(X))$ drops from nominal $90.0\%$ to $84.2\%$–$88.7\%$.

| Held-Out Target Cohort | Nominal Coverage | Source-Calib. Coverage | Target-Calib. Coverage | Acceptance Rate | Accuracy Gain under Abstention |
| :--- | :---: | :---: | :---: | :---: | :---: |
| UCI Hungarian ($N=293$) | 90.0% | 88.7% | 90.4% | 78.2% | +8.36% |
| UCI VA + Switz. ($N=308$) | 90.0% | 86.4% | 90.3% | 72.7% | +9.45% |
| UCI Cleveland Dedup ($N=303$) | 90.0% | 88.1% | 90.1% | 78.9% | +5.06% |
| Z-Alizadeh Sani ($N=303$) | 90.0% | 84.2% | 90.1% | 36.6% | +9.36% |
| Framingham Study ($N=4,238$) | 90.0% | 87.3% | 90.0% | 64.1% | -0.09% |

Crucially, on the Framingham cohort, conformal abstention does not improve classification accuracy ($51.32\%$ full vs. $51.23\%$ accepted, $\Delta = -0.09\%$). Because Framingham tracks 10-year incident events rather than cross-sectional angiographic CAD, the prediction score ranking is less reliable, causing abstention to discard both correct and incorrect predictions equally.

### 6.4 Test-Time Missingness vs. Native-NaN Trees (Table 6)
Prior studies claimed a $+0.15$ AUROC advantage over tree models under test-time missingness, but evaluated against median-imputed trees. In Table 6, we evaluate SchemaFlexTabNet-QC against **Native-NaN LightGBM** and **Native-NaN XGBoost**, which natively route missing values through optimal split paths.

| Missing Feature Rate | SchemaFlexTabNet-QC | Native-NaN LightGBM | Native-NaN XGBoost | Median-Imputed LightGBM | Advantage vs. Native LightGBM |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0% | **0.8520** | 0.7297 | 0.7265 | 0.7274 | +0.1223 |
| 10% | **0.8300** | 0.7035 | 0.7051 | 0.7090 | +0.1265 |
| 20% | **0.8036** | 0.6854 | 0.6940 | 0.6944 | +0.1182 |
| 30% | **0.7510** | 0.6718 | 0.6822 | 0.6779 | +0.0792 |
| 40% | **0.7159** | 0.6412 | 0.6575 | 0.6539 | +0.0747 |
| 50% | **0.7123** | 0.6131 | 0.6341 | 0.6337 | **+0.0992** |

SchemaFlexTabNet-QC degrades gracefully from $0.8520$ to $0.7123$ AUROC at $50\%$ missingness. Under $50\%$ missingness, the advantage over Native-NaN LightGBM is $+0.0992$ AUROC (and $+0.0782$ over Native-NaN XGBoost).

### 6.5 Architectural Component Ablation (Table 7)
Evaluated on the held-out multi-cohort test partition, Table 7 traces the contribution of each design component.

| Architecture Variant | Test AUROC | Accuracy (%) | Brier Score | Calibrated ECE | Monotonicity Violations (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Inner-Join Baseline | 0.6342 | 59.73% | 0.2532 | 0.1368 | 15.04% |
| 2. + Schema-Flexible Tokenizer | 0.8560 | 75.77% | 0.1872 | 0.1672 | 19.23% |
| 3. + Feature Dropout ($p=0.20$) | 0.8604 | 77.13% | 0.1540 | 0.1008 | 4.18% |
| 4. + Concept Head ($\lambda_1=0.10$) | 0.8611 | 77.82% | 0.1582 | 0.1081 | 12.97% |
| 5. **Full SchemaFlexTabNet-QC ($\lambda_2=1.20$)** | **0.8652** | **80.20%** | **0.1422** | **0.0572** | **0.26%** |

Adding the schema-flexible tokenizer drives the primary gain ($0.6342 \to 0.8560$). The auxiliary concept bottleneck provides only $+0.0007$ AUROC while increasing unregularized monotonicity violations to $12.97\%$. Introducing the positive-constrained skip path and autograd penalty ($\lambda_2 = 1.20$) suppresses violations to **$0.26\%$** while optimizing calibrated ECE to $0.0572$.

### 6.6 Mondrian Conformal Fairness Across Sex (Table 8)
Table 8 presents demographic subgroup calibration on the held-out test split. Uncalibrated predictions exhibit extreme false-positive rates ($\text{FPR} \approx 0.63$). Marginal split conformal prediction introduces a severe disparity: Male $\text{FPR} = 0.0375$ vs. Female $\text{FPR} = 0.1864$ (gap $= 0.1489$).

| Conformal Method | Subgroup | Acceptance Rate (%) | AUROC | Accuracy (%) | False Positive Rate (FPR) | False Negative Rate (FNR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Uncalibrated Base ($t=0.50$) | Male | 100.0% | 0.8078 | 63.81% | 0.6337 | 0.1101 |
| | Female | 100.0% | 0.7804 | 46.59% | 0.6338 | 0.1176 |
| 2. Marginal Split Conformal | Male | 76.7% | 0.8181 | 78.26% | 0.0375 | 0.3951 |
| | Female | 79.5% | 0.7982 | 78.57% | 0.1864 | 0.3636 |
| 3. **Mondrian Group-Fair** | Male | **79.5%** | **0.8417** | **78.44%** | **0.1733** | **0.2500** |
| | Female | **80.7%** | **0.7982** | **76.06%** | **0.2456** | **0.2143** |

Mondrian calibration adjusts subgroup operating thresholds, yielding Male $\text{FPR} = 0.1733$ and Female $\text{FPR} = 0.2456$ (gap $= 0.0723$). This **halves the false-positive disparity** ($0.1489 \to 0.0723$, a $51.4\%$ reduction), though it does not eliminate it entirely.

### 6.7 Cross-Hospital Federated Learning Simulation (Table 9)
Table 9 benchmarks Schema-Aware Masked FedAvg over 10 communication rounds across four hospital nodes with non-overlapping schemas.

| Deployment Framework | Privacy Mechanism | Communication Rounds | Held-Out Zero-Shot AUROC | Gap to Centralized |
| :--- | :--- | :---: | :---: | :---: |
| 1. Isolated Local Hospital Models | Local Data Only (No Sharing) | 0 | 0.7461 | -0.1285 |
| 2. **Schema-Aware FedAvg** | Masked Token Weights Only | **10** | **0.7671** | **-0.1075** |
| 3. Centralized Pooled Training | Full EHR Sharing (Pooled Data) | N/A | 0.8746 | 0.0000 |

Federation improves AUROC from $0.7461$ (isolated) to $0.7671$, closing **$16.3\%$ of the performance gap** to centralized pooling ($0.8746$). We note that while Schema-FedAvg shares only token weights, it provides no formal differential privacy ($\epsilon, \delta$) or cryptographic secure aggregation guarantees.

---

## 7. Discussion & Clinical Limitations

### 7.1 Outcome Construct Divergence
A vital methodological finding of this study is that multi-cohort cardiovascular datasets mix fundamentally distinct clinical constructs:
- **Prevalent CVD (Sulianova):** General population survey based on clinical history.
- **Angiographic CAD (UCI, Z-Alizadeh Sani):** Anatomical coronary stenosis ($\ge 50\%$) verified via fluoroscopy.
- **10-Year Incident CHD (Framingham):** Future occurrence of myocardial infarction or fatal coronary disease in longitudinal follow-up [17].

Pooling these cohorts under a single binary label $Y \in \{0, 1\}$ introduces construct heterogeneity. This explains why model performance on Framingham is uniformly lower ($0.6778$ AUROC) and why conformal abstention does not improve Framingham accuracy. Future multi-center clinical AI must employ multi-task heads or cohort-specific loss functions rather than assuming clinical interchangeability.

### 7.2 The 92% Dominance of Population Screening Records
The Sulianova cohort accounts for $68,568$ of the $74,013$ clean records ($92.6\%$). Unweighted pooled training biases representation learning toward non-invasive population vitals. While inverse-frequency cohort weighting balances gradient updates, it can increase gradient variance. Tabular foundation models must address cohort imbalance explicitly.

---

## 8. Conclusion & Open Science

This study proves that recent $>90\%$ cardiovascular risk stratification claims on standard benchmarks are not explainable by duplicate contamination alone, but reflect reporting contradictions in published confusion matrices and an inability to reproduce such performance under standard supervised learning ($73.48\%$ empirical ceiling). Furthermore, we demonstrate that standard inner-join merges discard vital clinical biomarkers, causing out-of-cohort performance to collapse.

To resolve these challenges, we introduced **SchemaFlexTabNet-QC**, a schema-flexible tabular transformer that operates across arbitrary feature sets, enforces clinical monotonicity ($<0.5\%$ violations), provides temperature-scaled conformal abstention, and enables cross-hospital federated training. Rigorous evaluation against Union-Schema LightGBM demonstrates that while deep tabular transformers offer advantages under high missingness ($+0.099$ AUROC at 50% dropout), properly tuned tree models remain competitive on static tabular data. Open science and forensic auditing are essential to ensure that clinical AI models deployed for cardiovascular patient care are statistically reliable, clinically valid, and reproducible.

The complete code, reproduction pipelines, and pre-trained weights are publicly available under the MIT License at:
`https://github.com/RathanPaida/SchemaFlexTabNet-QC`

---

## References

1. World Health Organization, "Cardiovascular diseases (CVDs) fact sheet," Tech. Rep., 2021.
2. M. Mahmud, S. K. Das, M. A. Kashem, and M. M. Rahman, "A clinical decision support system using deep tabular learning and quality control for cardiovascular disease prediction," *IEEE Open J. Comput. Soc.*, vol. 6, pp. 112–124, 2025.
3. S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, p. 100804, 2023.
4. Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, "Revisiting deep learning models for tabular data," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 34, 2021, pp. 18932–18943.
5. S. Ö. Arik and T. Pfister, "TabNet: Attentive interpretable tabular learning," in *Proc. AAAI Conf. Artif. Intell.*, vol. 35, no. 8, 2021, pp. 6679–6687.
6. B. Barz and J. Denzler, "Do we train on test data? Purging CIFAR of potential leaks," in *Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV)*, 2020, pp. 3010–3019.
7. R. Detrano *et al.*, "International application of a new probability algorithm for the diagnosis of coronary artery disease," *Am. J. Cardiol.*, vol. 64, no. 5, pp. 304–310, 1989.
8. V. Vovk, A. Gammerman, and G. Shafer, *Algorithmic Learning in a Random World*. New York, NY: Springer, 2005.
9. A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," *Found. Trends Mach. Learn.*, vol. 16, no. 4, pp. 494–591, 2023.
10. C. Gupta, A. Podkopaev, and A. Ramdas, "Distribution-free binary classification: Prediction sets, confidence intervals and calibration," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 34, 2021, pp. 3711–3723.
11. R. J. Tibshirani, R. F. Barber, E. Candes, and A. Ramdas, "Conformal prediction under covariate shift," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019.
12. R. F. Barber, E. J. Candes, A. Ramdas, and R. J. Tibshirani, "Predictive inference with the jackknife+," *Ann. Statist.*, vol. 49, no. 1, pp. 486–507, 2021.
13. E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, "Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach," *Biometrics*, vol. 44, no. 3, pp. 837–845, 1988.
14. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," in *Proc. Int. Conf. Artif. Intell. Statist. (AISTATS)*, 2017, pp. 1273–1282.
15. G. S. Collins *et al.*, "Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: TRIPOD+AI statement," *BMJ*, vol. 385, p. e078378, 2024.
16. R. Alizadehsani *et al.*, "A database of coronary artery disease patients based on angiography," *Comput. Methods Programs Biomed.*, vol. 111, no. 1, pp. 52–61, 2013.
17. T. R. Dawber, G. F. Meadors, and F. E. Moore, "Epidemiological approaches to heart disease: the Framingham Study," *Am. J. Public Health*, vol. 41, no. 3, pp. 279–281, 1951.
18. Z. Janosi, W. Steinbrunn, K. Pfisterer, and R. Detrano, "Hungarian Institute of Cardiology dataset," UCI Machine Learning Repository, 1988.
19. M. Pfisterer, W. Steinbrunn, and R. Detrano, "Heart disease dataset: Switzerland and Long Beach VA medical centers," UCI Machine Learning Repository, 1988.
20. G. Somepalli, M. Goldblum, A. Bollapragada, C. B. Bruss, and T. Goldstein, "SAINT: Improved neural networks for tabular data via row and column attention," *arXiv preprint arXiv:2106.01342*, 2021.
21. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 1321–1330.
22. A. J. Vickers and E. B. Elkin, "Decision curve analysis: A novel method for evaluating prediction models," *Med. Decis. Making*, vol. 26, no. 6, pp. 565–574, 2006.
