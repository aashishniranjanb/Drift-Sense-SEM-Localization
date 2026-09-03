# DRIFT-SENSE++ COMPLETE FAILURE DECOMPOSITION & RETRIEVAL CEILING AUDIT

**Target:** Audit all 180 Phase 2 development pairs (140 present, 40 absent) across the raw candidate universe to isolate the true systemic bottleneck and determine the mathematical score ceiling.

**Execution Mode:** Read-only / Diagnostic. **Zero production code was modified.**

---

## 1. Executive Summary & Core Discovery

1. **Rejection is NOT the Primary Bottleneck:**
   - Out of 64 currently rejected present pairs, **only 3 pairs** are pure rejection failures (where the top-1 candidate was $\le 5$ px but got rejected by the $V28\text{-C}$ gate).
   - Under the frozen candidate engine, the **Pure Rejection Ceiling is strictly 8.451 / 15.000** (a maximum gain of **+0.423 points** above baseline).
   - Therefore, spending more time on threshold tuning or rejection classification alone is mathematically capped at ~91.0 total.
2. **The Real Bottleneck is Periodic-Replica Ranking Confusion (18.6% of present pairs):**
   - In **26 pairs** (13 in Set A, 13 in Set B), the candidate extraction engine **successfully retrieved the true target ($\le 5$ px) into the 200-candidate pool**, with an average error of **1.90 px** (and $\le 1.0$ px in 14 cases).
   - However, a near-identical periodic replica beat the true target to Rank 1.
   - Solving candidate re-ranking (Truth vs. Replica disambiguation) unlocks **+2.40 points in rejection**, boosting the single-file score to **93.44 / 100.00**.
3. **The Retrieval Ceiling (25.0% of present pairs):**
   - In **35 pairs**, no candidate in the top-200 pool was within 5 px. However, **22 of those 35 pairs are within 5–10 px** (just 1 periodic pitch off).
   - A periodic lattice rescue step (e.g. R2 grid expansion) would capture these 22 near-retrievals, pushing rejection to **13.2+ points** and total score to **96.2+**.
4. **Near-Tie Discrimination Reality:**
   - Among the 76 successfully accepted pairs, the median margin between GT and the strongest competitor is only **0.0027** ($< 0.01$ in 60 out of 76 cases).
   - This proves that true matches and periodic replicas are virtually indistinguishable by raw correlation alone; they require multi-signal structural context and constellation consistency to separate.

---

## 2. Comprehensive 4-Way Failure Decomposition (140 Present Pairs)

Every present pair was audited by extracting the full top-200 candidate pool using the exact V25 pose parameters:

| Failure Category | Set A | Set B | Total Pairs | % of Present | Mechanism & Root Cause |
|---|---|---|---|---|---|
| **SUCCESS_ACCEPTED** | 40 | 36 | **76** | **54.3%** | Top-1 is correct ($\le 5$px) and accepted ($V28\text{-C} > 0.873$). 63 pairs are subpixel ($\le 1$px). |
| **REJECTION_FAILURE (Type 1)** | 1 | 2 | **3** | **2.1%** | Top-1 is correct ($\le 5$px), but conservatively rejected by $V28\text{-C}$. (`pair_027`, `pair_078`, `pair_116`). |
| **RANKING_FAILURE (Type 2)** | 13 | 13 | **26** | **18.6%** | **GT candidate is in top-200 pool** (avg error 1.90px), but lost Rank 1 to a periodic replica! |
| **RETRIEVAL_FAILURE (Type 3)** | 16 | 19 | **35** | **25.0%** | GT candidate did not enter top-200 pool (22 are near-misses within 5–10px; 8 are 10–25px; 5 are >25px). |
| **TOTAL PRESENT** | **70** | **70** | **140** | **100.0%** | Complete accounting of all present instances. |

*Note on Absent Pairs (Set C, $n=40$):*
- 38 pairs are correctly rejected ($TP=38$).
- 2 pairs are false accepts (`pair_140`, `pair_159`, $FN=2$).

---

## 3. GT vs Best-Replica Margin & Difficulty Distribution

Evaluating the margin $\Delta = \text{Top1} - \text{Top2}$ across populations:

1. **Successful Acceptances ($n=76$):**
   - $\Delta > 0.10$: 0 pairs (0.0%)
   - $0.05 < \Delta \le 0.10$: 0 pairs (0.0%)
   - $0.01 < \Delta \le 0.05$: 16 pairs (21.1%)
   - $\Delta \le 0.01$: **60 pairs (78.9%)**
   - **Median victory margin:** **0.0027** (A razor-thin 0.27% difference!).
2. **Ranking Failures ($n=26$):**
   - The periodic replica won Rank 1 with a median margin over Top-2 of **0.0464**.
   - The true candidate sits at Ranks 2–15 in the pool.
3. **Retrieval Failures ($n=35$):**
   - $5 < \text{Min Error} \le 10$ px: **22 pairs** (62.9% of retrieval failures are 1 lattice pitch away).
   - $10 < \text{Min Error} \le 25$ px: **8 pairs** (22.9%).
   - $\text{Min Error} > 25$ px: **5 pairs** (14.2%).

---

## 4. Rigorous Mathematical Ceilings Comparison

| Pipeline Configuration | Localized $\le 5$px | Localization Pts | Rejection F1 | Rejection Pts | Pose Pts | Total Score (excl. RGB) | Score with +6.0 RGB |
|---|---|---|---|---|---|---|---|
| **Current Locked Baseline** | 76 / 140 (54.3%) | **40.00** | 0.5352 | **8.03** | 19.74 | **90.98** | **96.98** |
| **Ceiling 1: Pure Rejection Oracle** | 79 / 140 (56.4%) | **40.00** | 0.5634 | **8.45** (+0.42) | 19.74 | **91.46** | **97.46** |
| **Ceiling 2: Top-200 Pool Re-Ranker** | 105 / 140 (75.0%) | **40.00** | 0.6957 | **10.43** (+2.40) | 19.74 | **93.44** | **99.44** |
| **Ceiling 3: Full Retrieval + Re-Ranker**| 127 / 140 (90.7%) | **40.00** | 0.8812 | **13.22** (+5.19) | 19.74 | **96.23** | **102.23** |

### Insights from the Ceiling Ladder:
- **Pure Rejection tuning is a dead end:** You can only gain at most **+0.42 points** because the other 61 rejected candidates are periodic replicas $> 5$ px away.
- **The Top-200 Pool Re-Ranker is the highest-ROI move:** In 26 pairs, the true candidate is already inside the candidate pool. Promoting the true candidate over the periodic replica unlocks **+2.40 points** and pushes the grayscale total to **~93.44**.
- **RGB Bonus:** With the +6.00 RGB bonus, our current baseline is already effectively at **~96.98**, and a re-ranker pushes it to **~99.44**.

---

## 5. The Path to Championship Execution (Next Concrete Steps)

Now that the failure decomposition is 100% transparent:

1. **Stop Rejection Thresholding:** Do not modify $V28\text{-C}$ or train rejection classifiers on the top-1 candidate alone.
2. **Build the Candidate Disambiguation Re-Ranker (Truth vs. Replica):**
   - For each candidate in the top-200 pool, compute pairwise relative features against its nearest competitors:
     - $\Delta \text{NCC}$, $\Delta \text{context}$, $\Delta \text{gradient}$, $\Delta \text{phase}$
     - Local constellation density and row/column lattice residual
   - Train on synthetic periodic hard-negatives to promote the true candidate to Rank 1.
   - Run in shadow mode on the 26 Ranking Failure pairs.
3. **Preserve All Locked Components:**
   - Pose: Keep $V54$ scale-only quadratic refinement (Scale MAE 0.0352).
   - Localization: Keep the 40/40 anchor invariant.
   - Calibration: Keep `calib_lean.pkl` (AUC 0.9953).
