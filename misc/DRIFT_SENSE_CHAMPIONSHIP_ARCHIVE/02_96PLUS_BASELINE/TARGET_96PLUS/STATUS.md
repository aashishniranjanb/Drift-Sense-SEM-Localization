# TARGET_96PLUS — Grayscale Optimization Target Status

**Status:** OPTIMIZATION TARGET (UNVERIFIED)
**Current Baseline:** V54 Scale Refined Golden Anchor @ 91.040 / 100
**Target Benchmark Score:** ≥ 96.000 / 100

> [!NOTE]
> This folder is designated as `TARGET_96PLUS` until an empirical clean-room benchmark run produces an actual measured score ≥ 96.000. It must NOT be named `96PLUS_VERIFIED` until such verification is established.

---

## Target Score Breakdown (Grayscale Baseline vs 96+ Goal)

| Component | Max Points | Current Verified (V54 Anchor) | Target 96+ Goal | Gap / Requirement |
|---|---|---|---|---|
| **Localization** | 40.00 | 40.00 | 40.00 | Anchor protected (100% SetA/SetB ≤5px) |
| **Pose Estimation** | 20.00 | 19.74 | 19.80 | Scale MAE = 0.0352 (V54 quadratic scale refinement) |
| **Rejection** | 15.00 | 8.03 | 13.00 | Move F1 from 0.543 → ~0.87 (Requires addressing ranking/retrieval failures) |
| **Calibration** | 10.00 | 8.27 | 9.20 | Lean ROC-AUC = 0.9953 |
| **Efficiency** | 5.00 | 5.00 | 5.00 | Full pipeline execution < 2.0s / pair |
| **Documentation** | 10.00 | 10.00 | 10.00 | Complete audit trails & reproducible code |
| **TOTAL** | **100.00** | **91.040** | **96.040+** | **+5.00 points required** |

---

## Bottleneck & Failure Analysis

To move from 91.040 → 96.000+, the candidate engine must address the remaining 61 non-perfect cases in the 180-pair dev benchmark:
1. **26 Ranking Failures:** Correct candidate is inside top-200 pool, but periodic replica wins rank 1.
2. **35 Retrieval Failures:** Correct candidate is outside top-200 pool (22 near-misses 5–10px away).

---

## Optimization Tree

```
91.040 VERIFIED GOLDEN
        │
        ├── Grayscale candidate retrieval/reranking
        │        ↓
        │     96+ TARGET (TARGET_96PLUS)
        │
        └── RGB Extension
                 ↓
          +6 BONUS ELIGIBILITY
                 ↓
             102+ TARGET (TARGET_102PLUS)
```
