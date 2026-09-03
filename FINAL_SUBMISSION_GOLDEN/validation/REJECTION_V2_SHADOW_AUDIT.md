# REJECTION_V2 SHADOW-ONLY AUDIT & THEORETICAL CEILING REPORT

**Environment & Invariants:**
- Production Engine: FROZEN (`V25` localization, `V28-C` presence gate, `V39` pose, `V54` scale-only quadratic refinement, `calib_lean.pkl`).
- Production code modified: **NONE (0 lines changed in production).**
- Execution Mode: **SHADOW DIAGNOSTIC ONLY.**
- Benchmark Data: `data/phase2_dev/pairs.csv` (180 pairs: 140 present, 40 absent).

---

## 1. Executive Summary & Core Discovery

We conducted the empirical shadow audit and theoretical ceiling analysis for Rejection V2. The findings definitively settle the architecture roadmap:

1. **Theoretical Rejection Ceiling is Hard-Capped at 8.451 / 15.000 (+0.423 max delta):**
   - Out of the 64 currently rejected present pairs (FP in rejection taxonomy), **62 pairs have top-1 V25 localization errors > 5.0 px** (averaging 70–250 px, corresponding to periodic replica grid hops).
   - Only **2 pairs** (`pair_027` with 3.54 px error and `pair_078` with 0.29 px error) have raw candidates within $\le 5$ px.
   - Therefore, any classifier attempting to "rescue" candidates from the rejected pool without changing the candidate retrieval/pool engine will inevitably accept false periodic replicas, directly degrading the pristine 40.00/40.00 localization score.
2. **Accepted-Candidate Veto Destroys Score:**
   - Any veto threshold that removes false accepts also accidentally throws away legitimate, low-contrast present matches. At $T_{veto} = 0.01$, 26 true matches are discarded for every 1 false accept eliminated, producing a net loss of **-1.105 points**.
3. **Rejected-Candidate Rescue Destroys Localization:**
   - Even at an ultra-conservative threshold of $T_{rescue} = 0.99$, 12 candidates are rescued—**all 12 are wrong periodic replicas (0 correct, 12 wrong)**. Localization immediately crashes from 40.00 to 37.00 (-3.00 points).
4. **Final Recommendation:** **DO NOT PROMOTE.** Rejection V2 must remain disabled. Rejection cannot carry the team to 96+ under a frozen candidate engine.

---

## 2. Current Baseline Confusion Matrix (180 Pairs)

Evaluating under the competition convention where **Absent ($gt\_found = 0$) is the Positive class for Rejection**:

| Metric | Value | Description |
|---|---|---|
| **TP (Absent Correctly Rejected)** | **38 / 40** | Only 2 absent pairs are incorrectly accepted (`pair_140`, `pair_159`). |
| **FN (Absent Incorrectly Accepted)** | **2 / 40** | The only 2 genuine false accepts in the entire dataset. |
| **FP (Present Incorrectly Rejected)** | **64 / 140** | 62 are periodic replicas (>5px), 2 are recoverable ($\le 5$px). |
| **TN (Present Correctly Accepted)** | **76 / 140** | 100% of accepted present pairs have $\le 5$px error (63 are $\le 1$px). |
| **Precision** | **0.3725** | $38 / (38 + 64)$ |
| **Recall** | **0.9500** | $38 / (38 + 2)$ |
| **F1 Score** | **0.5352** | $2 \times \frac{0.3725 \times 0.9500}{0.3725 + 0.9500}$ |
| **Official Rejection Score** | **8.028 / 15.000** | $0.5352 \times 15.0$ |

---

## 3. Mathematical Rejection Ceiling (Frozen Candidate Engine)

Assuming an omniscient oracle that perfectly solves the rejection classification problem **without modifying the V25 candidate coordinates**:

- **Maximum Possible TP:** 40 / 40 (both `pair_140` and `pair_159` rejected).
- **Minimum Possible FN:** 0 / 40.
- **Minimum Possible FP:** 62 / 140 (because rescuing any of the 62 periodic replicas injects $>5$ px error and breaks the 40/40 localization score).
- **Maximum Possible TN:** 78 / 140 (rescuing `pair_027` and `pair_078` only).

$$\text{Precision}_{max} = \frac{40}{40 + 62} = 0.39216$$
$$\text{Recall}_{max} = \frac{40}{40} = 1.00000$$
$$\text{F1}_{max} = \frac{2 \times 0.39216 \times 1.0}{1.39216} = 0.56338$$
$$\text{Max Rejection Points} = 0.56338 \times 15.0 = \mathbf{8.451 \text{ points}}$$

$$\Delta_{\text{ceiling}} = 8.451 - 8.028 = \mathbf{+0.423 \text{ points}}$$

> [!IMPORTANT]
> **Key Conclusion:** The theoretical headroom in rejection under the frozen candidate engine is at most **+0.42 points**, NOT +4 to +6 points. 96+ cannot mathematically be achieved by tuning rejection alone.

---

## 4. Synthetic Training & Model Evaluation

Per instructions, the model was trained exclusively on a 20,000-sample synthetic dataset (70% train / 30% held-out validation) featuring 4 distinct classes:
1. **Class A (5,000):** Genuine present instances with semiconductor degradations.
2. **Class B (5,000):** Periodic-replica hard negatives with replica NCC $\approx$ GT NCC.
3. **Class C (5,000):** Absent architecture negatives.
4. **Class D (5,000):** Near-miss localization cases ($>5$ px error on the same periodic pitch).

### Model Performance on Synthetic Held-Out Test Set (6,000 samples)
- **Logistic Regression:** ROC AUC = 1.0000, PR AUC = 1.0000, Brier = 0.0001
- **HistGradientBoosting (depth=2):** ROC AUC = 1.0000, PR AUC = 1.0000, Brier = 0.0001

### Model Evaluation on 180-Pair Dev Set (Shadow Diagnostic)
- **Logistic Regression $P(\text{correct})$ AUC:** **0.7354**
- **HistGradientBoosting $P(\text{correct})$ AUC:** **0.6961**

Score distribution across true populations (HGB):
- Correct Accepted (True Positives, $n=76$): mean = 0.571 (min 0.000, max 1.000)
- Incorrect Accepted (False Accepts, $n=2$): mean = 0.496 (min 0.002, max 0.989)
- Correct Rejected (Recoverable FN, $n=2$): mean = 0.238 (min 0.000, max 0.715)
- Incorrect Rejected (Absent + Periodic Replicas, $n=100$): mean = 0.267 (min 0.000, max 1.000)

---

## 5. Shadow Intervention Simulations (No Code Modified)

### A. Accepted-Candidate Veto Sweep (Targeting the 2 False Accepts)
Evaluating whether low $P(\text{correct})$ can safely veto `pair_140` and `pair_159` without harming genuine matches:

| $T_{veto}$ | Candidates Vetoed | Correct Matches Removed | False Accepts Removed | Loc Points | Rej Points | $\Delta$ Rejection | $\Delta$ Total |
|---|---|---|---|---|---|---|---|
| **0.00 (Base)** | 0 | 0 | 0 | 40.00 | 8.028 | 0.000 | 0.000 |
| **0.01** | 27 | 26 | 1 | 40.00 | 6.923 | -1.105 | **-1.105** |
| **0.02** | 28 | 27 | 1 | 40.00 | 6.882 | -1.146 | **-1.146** |
| **0.05** | 28 | 27 | 1 | 40.00 | 6.882 | -1.146 | **-1.146** |
| **0.10** | 28 | 27 | 1 | 40.00 | 6.882 | -1.146 | **-1.146** |
| **0.20** | 28 | 27 | 1 | 40.00 | 6.882 | -1.146 | **-1.146** |

*Analysis:* Vetoing accepted candidates is catastrophic. For every 1 false accept pruned, 26 true localizations are erroneously thrown away, collapsing rejection score by over 1.1 points.

---

### B. Rejected-Candidate Rescue Sweep (Targeting the 64 Rejected Pairs)
Evaluating whether high $P(\text{correct})$ can safely rescue present pairs without accepting periodic replicas:

| $T_{rescue}$ | Candidates Rescued | Correct Rescued ($\le 5$px) | Wrong Rescued ($>5$px) | Loc Points | Rej Points | $\Delta$ Loc | $\Delta$ Rej | $\Delta$ Total |
|---|---|---|---|---|---|---|---|---|
| **Base** | 0 | 0 | 0 | 40.00 | 8.028 | 0.00 | 0.000 | 0.000 |
| **0.80** | 21 | **0** | **21** | 35.33 | 7.686 | -4.67 | -0.342 | **-5.009** |
| **0.85** | 21 | **0** | **21** | 35.33 | 7.686 | -4.67 | -0.342 | **-5.009** |
| **0.90** | 21 | **0** | **21** | 35.33 | 7.686 | -4.67 | -0.342 | **-5.009** |
| **0.95** | 19 | **0** | **19** | 35.58 | 7.805 | -4.42 | -0.223 | **-4.638** |
| **0.97** | 18 | **0** | **18** | 35.85 | 7.742 | -4.15 | -0.286 | **-4.440** |
| **0.98** | 18 | **0** | **18** | 35.85 | 7.742 | -4.15 | -0.286 | **-4.440** |
| **0.99** | 12 | **0** | **12** | 37.00 | 7.846 | -3.00 | -0.182 | **-3.182** |

*Analysis:* Rescuing rejected candidates is universally destructive. Even at $T_{rescue} = 0.99$, **0 correct candidates are rescued, while 12 false periodic replicas are accepted**, causing a catastrophic 3.00 point drop in localization.

---

## 6. Required Deliverables Summary

1. **Best Safe Veto Threshold:** **NONE** ($T_{veto} = 0.00$ / Disabled). Any $T_{veto} > 0$ reduces total score.
2. **Best Safe Rescue Threshold:** **NONE** ($T_{rescue} = 1.00$ / Disabled). Any $T_{rescue} < 1.0$ reduces total score.
3. **Expected Rejection Improvement:** **0.000 points** (attempting rejection promotion yields negative net points).
4. **Expected Total Improvement:** **0.000 points.**
5. **Theoretical Rejection Ceiling:** **8.451 points** ($\Delta_{max} = +0.423$ points above baseline).
6. **FINAL DECISION:** **DO NOT PROMOTE.** 

---

## 7. Strategic Implication for 96+ Score

Because the theoretical ceiling of rejection under the current candidate engine is strictly **8.451 points**, spending further engineering cycles on rejection classification is mathematically a dead end for reaching 96+.

The path to 96+ must instead focus on:
1. **Candidate Pool Retrieval (Replica Disambiguation):** Re-ranking the 200 candidates inside the pool so that the true $\le 5$ px candidate replaces the periodic replica *before* the rejection gate.
2. **Pose Refinement:** Continuing the isolated parabolic tuning (which already gained +31% scale precision).
3. **The +6.00 RGB Bonus:** Ensuring full compliance with the organizer's RGB bonus specification.
