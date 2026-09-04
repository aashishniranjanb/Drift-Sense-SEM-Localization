# RERANK-V2 SHADOW EXPERIMENT AUDIT REPORT

**Objective:** Build and evaluate a pairwise Truth-vs-Replica re-ranker (RERANK-V2) in **strict shadow mode** on the top-200 candidate pool to attempt to recover the 26 known ranking failures.

**Execution Rule:** **ZERO production code modified.** Golden baseline remains 100% untouched.

---

## 1. Executive Summary & Required Output

1. **Best Model:** **None.** (Both Logistic Regression and HistGradientBoosting depth=2/3 failed the mandatory safety gate).
2. **Best Margin Threshold:** **$T = 0.005$** (Minimum damage threshold, but still negative net yield).
3. **26-Case Ranking Recoveries:** **0 / 26 recovered.** (Because the true candidate sits at median rank 22.5 in the raw candidate pool, evaluating only the top-10 candidates excluded the true match from the pairwise tournament).
4. **76-Case Regressions (Successful Pairs Broken):** **2 broken** at $T = 0.005$ (`pair_010`, `pair_131`); **38 broken** if applied unconditionally.
5. **Localization Score Change:** **40.00 $\to$ 38.94 (-1.06 points).**
6. **Rejection Score Change:** **8.028 $\to$ 8.028 (0.000 delta).**
7. **Pose Score Change:** **19.743 $\to$ 19.743 (0.000 delta).**
8. **Calibration Score Change:** **8.269 $\to$ 8.012 (Spearman drops -0.026).**
9. **Runtime Change:** $+1.2$ s/pair if run across candidates.
10. **Total Score Change:** **91.040 $\to$ 89.979 (-1.061 points).**
11. **FINAL DECISION:** **DO NOT PROMOTE.** (Strict safety gate triggered: breaking any $\le 5$ px localization without positive net official score mandates immediate rejection).

---

## 2. Methodology & Synthetic Adversarial Training (Step 4 & 5)

Per instructions, 10,000 synthetic pairwise instances were generated using deterministic random seeds.
- **Adversarial Hard Examples (55% of distribution):** Instances where $\text{NCC}(\text{replica}) > \text{NCC}(\text{true})$ by 0.005 to 0.045, with close context and gradient signatures.
- **Standard Examples (45%):** GT slightly ahead across all signals.

### Model Performance on Held-Out Synthetic Validation Set:
- **Model A (LogisticRegression):** Held-Out AUC = 1.0000, Brier = 0.0039
- **Model B (HistGradientBoosting depth=2):** Held-Out AUC = 1.0000, Brier = 0.0000
- **Model C (HistGradientBoosting depth=3):** Held-Out AUC = 1.0000, Brier = 0.0000

### Learned Logistic Regression Weights:
- $d_{\text{lattice\_res}}$: **-13.8451**
- $d_{\text{ctx}}$: **+6.2969**
- $d_{\text{sharpness}}$: **+5.5553**
- $d_{\text{phase}}$: **-4.9197**
- $d_{\text{neigh}}$: **+4.5957**
- $d_{\text{prominence}}$: **+3.0199**
- $d_{\text{corr}}$: **-2.2161**

---

## 3. The Forensic Root Cause of Failure

Why did the pairwise ML re-ranker fail on the real 180-pair benchmark despite perfect synthetic performance?

1. **Candidate Extraction Depth Mismatch:**
   - In the 26 ranking failures, the true GT candidate is buried at **median rank 22.5** (up to rank 126) in the candidate pool.
   - Performing a top-10 candidate tournament failed because the true candidate was **not in the top 10** for 20 of the 26 pairs.
2. **Negative Correlation Weight Inversion:**
   - Because the synthetic training set over-penalized correlation ($d_{\text{corr}} = -2.2161$) to counter adversarial replicas, in the 76 true success pairs—where GT legitimately has higher correlation—the model penalized the correct candidate, causing **38 out of 76 successful pairs to break** under unconstrained re-ranking.
3. **Lattice Residual Brittleness:**
   - Real SEM images suffer from perspective drift, magnification non-uniformity, and scan distortion. The synthetic lattice residual ($d_{\text{lattice\_res}} = -13.84$) proved too brittle on real semiconductor images.

---

## 4. Second-Look Margin Policy Sweep (Step 8)

Evaluating the policy:
$$\text{If } \text{top1\_margin} > T: \text{ keep baseline winner; Else: apply RERANK-V2}$$

| Threshold $T$ | Rescued (out of 26) | Broken (out of 76) | Net Localization Gain | Localization Points | Total Score |
|---|---|---|---|---|---|
| **Baseline ($T=0.000$)** | **0** | **0** | **0** | **40.00** | **91.040** |
| **0.005** | 0 | 2 (`pair_010`, `pair_131`) | -2 | 38.94 | 89.979 |
| **0.010** | 0 | 3 | -3 | 38.41 | 89.448 |
| **0.020** | 0 | 3 | -3 | 38.41 | 89.448 |
| **0.030** | 0 | 3 | -3 | 38.41 | 89.448 |
| **0.050** | 0 | 4 | -4 | 37.89 | 88.917 |
| **0.075** | 0 | 7 | -7 | 36.31 | 87.324 |
| **0.100** | 0 | 12 | -12 | 33.68 | 84.669 |

**Conclusion:** At every tested threshold, the pairwise ML model causes negative net localization. 

---

## 5. Decision & Next Architectural Move

**RULE ENFORCED: DO NOT PROMOTE.**
The golden baseline (`FINAL_SUBMISSION_GOLDEN`) remains completely intact at **91.040** (Scale MAE = 0.0352).

### Key Takeaway for Championship Development:
- Training generic pairwise classifiers on synthetic data creates distribution drift where real SEM charging/distortion causes catastrophic inversions on verified matches.
- The only proven safe candidate recovery mechanism remains **constrained multi-signal second-look evaluation** (like the linear context-neighborhood boost from RERANK-V1), but it must operate deeper in the candidate pool (at depth $K=30-50$) rather than $K=10$.
- **Phase C (Lattice Retrieval Rescue for the 22 Near-Misses):** The 22 near-misses that sit 5–10 px outside the pool represent a direct geometric opportunity that does not depend on candidate pool re-ordering.
