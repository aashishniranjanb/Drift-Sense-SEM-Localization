# RERANK-V3 SHADOW EXPERIMENT AUDIT REPORT

**Objective:** Build and evaluate a constrained, non-ML structural candidate selector (RERANK-V3) across depths $K \in \{20, 30, 40, 50, 75\}$ in **strict shadow mode** to attempt to recover the 26 known ranking failures.

**Execution Rule:** **ZERO production code modified.** Golden baseline (`FINAL_SUBMISSION_GOLDEN/`) remains 100% untouched at **91.040**.

---

## 1. Executive Summary & Required Output

1. **Best K:** **None** ($K = 20, 30, 40, 50, 75$ evaluated).
2. **Best Thresholds:** **None** (Across 270 parameter combinations, zero configurations achieved the mandatory safety requirement of 0 regressions).
3. **26-Case Ranking Recoveries:** **0 / 26 recovered** safely. (Only 3 of the 26 failure pairs possess two simultaneous positive structural signals; in 23 of 26 pairs, the periodic replica leads in 2 or more structural metrics).
4. **76-Case Regressions:** **7 to 14 broken** across all active threshold combinations.
5. **Localization Score:** **40.000** (Golden Baseline maintained; shadow override dropped to 36.31).
6. **Rejection Score:** **8.028** (Golden Baseline maintained; shadow override dropped to 7.21).
7. **Pose Score:** **19.743** (Golden Baseline maintained, Scale MAE = 0.0352).
8. **Calibration AUC:** **0.9953** (Golden Baseline maintained).
9. **Runtime:** $0.09$ s/pair (Golden Baseline maintained).
10. **Total Score:** **91.040 / 100.00** (Golden Baseline maintained; shadow override dropped to 86.27).
11. **FINAL DECISION:** **DO NOT PROMOTE.** (The mandatory safety rule is triggered: any configuration that breaks verified $\le 5$ px localizations without positive net official score is immediately rejected).

---

## 2. Methodology & Multi-Signal Grid Sweep

All 102 target pairs (26 ranking failures + 76 verified successes) were processed in parallel across 8 CPU workers:
- Candidates were extracted to depth $K \in \{20, 30, 40, 50, 75\}$.
- For each candidate, relative features were computed: $\Delta \text{corr}$, $\Delta \text{ctx}$, $\Delta \text{neigh}$, $\Delta \text{grad}$, $\Delta \text{sharpness}$, $\Delta \text{prominence}$.
- The **2-out-of-3 Independent Signal Rule** was evaluated:
  - $\text{Condition 0: } \Delta \text{corr} \ge -\theta_{\text{corr}}$
  - $\text{Signal 1 (Context): } \Delta \text{ctx} \ge \theta_{\text{ctx}}$
  - $\text{Signal 2 (Neighborhood): } \Delta \text{neigh} \ge \theta_{\text{neigh}}$
  - $\text{Signal 3 (Sharpness/Grad): } \Delta \text{sharp} \ge \theta_{\text{sharp}} \lor \Delta \text{grad} \ge 0.010$

270 distinct threshold combinations were evaluated across all 5 candidate pool depths.

---

## 3. Deep Forensic Root Cause: The Multi-Signal Symmetry Trap

Why did RERANK-V3 fail to recover candidates while breaking 7 successes?

1. **Signal Correlation Asymmetry:**
   - Among the 26 ranking failures:
     - $\Delta \text{ctx} > 0$: **8 pairs** (30.8%)
     - $\Delta \text{neigh} > 0$: **7 pairs** (26.9%)
     - $\Delta \text{grad} > 0$: **0 pairs** (0.0% — the periodic replica has higher gradient NCC in 100% of failure cases!)
     - Both $\Delta \text{ctx} > 0$ AND $\Delta \text{neigh} > 0$: **ONLY 3 PAIRS** (`pair_043`, `pair_045`, `pair_061`).
   - In 23 of the 26 ranking failures, the true target is defeated by the replica in 2 or all 3 structural metrics because the nanoscale lattice repetition creates identical local context and gradient fields.
2. **False Challenger Breaches in the 76 Successes:**
   - In 7 of the 76 successful pairs, at least one of the 50 candidate peaks has a slightly higher context or neighborhood score than the true target due to edge noise or background texture.
   - When the threshold was opened enough to permit overrides, these false candidates replaced verified true matches, breaking 7 to 12 successful localizations.
3. **Net Yield:**
   - Maximum Rescued: 0
   - Minimum Broken: 7
   - Net Localization Gain: **-7** (Universal regression across all 270 combinations).

---

## 4. Top Grid Configurations Summary

| $K$ | $\theta_{\text{corr}}$ | $\theta_{\text{ctx}}$ | $\theta_{\text{neigh}}$ | $\theta_{\text{sharp}}$ | Rescued (of 26) | Broken (of 76) | Net Gain | Status |
|---|---|---|---|---|---|---|---|---|
| **20** | 0.020 | 0.020 | 0.035 | 0.040 | 0 | 7 | -7 | **REJECTED** |
| **20** | 0.020 | 0.035 | 0.020 | 0.020 | 0 | 7 | -7 | **REJECTED** |
| **30** | 0.020 | 0.020 | 0.035 | 0.040 | 0 | 7 | -7 | **REJECTED** |
| **30** | 0.020 | 0.035 | 0.020 | 0.020 | 0 | 7 | -7 | **REJECTED** |
| **40** | 0.020 | 0.035 | 0.035 | 0.040 | 0 | 7 | -7 | **REJECTED** |
| **50** | 0.020 | 0.035 | 0.035 | 0.040 | 0 | 7 | -7 | **REJECTED** |
| **75** | 0.020 | 0.035 | 0.035 | 0.040 | 0 | 7 | -7 | **REJECTED** |

---

## 5. Final 180-Pair Benchmark & Promotion Verdict

Because zero configurations passed the mandatory safety requirement ($\text{broken} = 0$), no override was applied to the benchmark. The golden baseline predictions are 100% preserved in `rerank_v3_shadow_predictions.csv`:

| Component | Golden Baseline | RERANK-V3 Shadow | Official Delta |
|---|---|---|---|
| **Localization (40)** | **40.000** | **40.000** | **+0.000** |
| **Rejection (15)** | **8.028** | **8.028** | **+0.000** |
| **Pose (20)** | **19.743** | **19.743** | **+0.000** |
| **Calibration (10)** | **8.269** | **8.269** | **+0.000** |
| **Efficiency (5)** | **5.000** | **5.000** | **+0.000** |
| **Documentation (10)** | **10.000** | **10.000** | **+0.000** |
| **TOTAL SCORE (100)** | **91.040** | **91.040** | **+0.000** |

**VERDICT: DO NOT PROMOTE.**
The golden baseline remains intact at **91.040**.

---

## 6. Strategic Takeaway for Phase 9 & 10 (Lattice Retrieval Rescue)

The empirical failure of both RERANK-V2 (ML pairwise) and RERANK-V3 (constrained heuristic) settles the ranking question:
- Re-ranking deep in the candidate pool ($K=20-75$) is an ill-conditioned problem because true targets and periodic replicas share the same underlying periodic structure; attempting to override correlation invariably causes regressions on verified matches.
- In contrast, **Phase 9/10: Local Lattice Retrieval Rescue** targets the **22 near-miss retrieval failures that are only 5–10 px outside the pool**. Probing the exact physical lattice pitch vectors ($x \pm v_x, y \pm v_y$) operates on pure physical geometry, avoiding the ranking ambiguity trap entirely.
