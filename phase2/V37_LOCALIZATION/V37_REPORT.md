# V37 Representation Stability Analysis Report

## Executive Summary
- **Total Pairs Analyzed**: 140
- **SAFE Candidates (loc_error <= 5.0px)**: 28 (20.0%)
- **WRONG Candidates (loc_error > 5.0px)**: 112 (80.0%)
- **Runtime**: 125.09 seconds

---

## Stability Metrics: SAFE vs WRONG Candidates

| Metric | SAFE (n=28) | WRONG (n=112) | Delta (SAFE - WRONG) |
|---|---|---|---|
| **Score Mean** | 0.0568 | 0.0330 | +0.0239 |
| **Score Std** | 0.0341 | 0.0291 | +0.0050 |
| **Rank Std** | 30.5344 | 32.2876 | -1.7532 |
| **Coordinate Std (px)** | 2.3350 | 2.0403 | +0.2948 |
| **Winner Frequency** | 0.0000 | 0.0000 | +0.0000 |
| **Representation Agreement (<=3)** | 0.00 / 6 | 0.06 / 6 | -0.06 |

---

## Per-Representation Rank 1 Frequency
- **Original**: SAFE rank=1 in 0.0% | WRONG rank=1 in 0.0%
- **Normalized**: SAFE rank=1 in 0.0% | WRONG rank=1 in 0.0%
- **High-pass**: SAFE rank=1 in 0.0% | WRONG rank=1 in 0.0%
- **Gradient**: SAFE rank=1 in 0.0% | WRONG rank=1 in 0.0%
- **Blur**: SAFE rank=1 in 0.0% | WRONG rank=1 in 0.0%
- **Sharp**: SAFE rank=1 in 0.0% | WRONG rank=1 in 0.0%

---

## Key Findings & Signal Assessment
1. **Geometric Stability**:
   - SAFE candidates show geometric coordinate std of `2.335px` vs `2.040px` for WRONG candidates.
2. **Representation Agreement**:
   - SAFE candidates remain top-ranked across `0.00` representations, compared to `0.06` for WRONG candidates.

---
*Report generated automatically by `v37_stability.py`*
