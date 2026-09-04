# Drift-Sense++ Score Integrity Statement

**Scientific Transparency & Benchmark Separation**

In competitive evaluations, maintaining rigorous boundaries between internal validation, development measurements, and official scores is critical. This document defines the exact scope of every reported metric.

```
                    BENCHMARK INTEGRITY TAXONOMY
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    ▼                             ▼                             ▼
OFFICIAL ORGANIZER SCORE     RELEASED DEV MEASUREMENT     HISTORICAL DIAGNOSTICS
[UNKNOWN UNTIL JURY RUN]        [90.50 / 100.00]         [SUPERSEDED EXPLORATIONS]
Evaluated on private         Computed strictly on the     Ablation sweeps, OOF
held-out test set by         released 180-pair dataset    tuning, research runs
Applied Materials.           per official rubric.         (V1 - V47). Not claimed.
```

---

## 1. Explicit Category Separation

### A. Official Organizer Score
- **Status:** **UNKNOWN.**
- **Scope:** Applied Materials will evaluate `register.py` against an undisclosed, held-out private test set on their 4-core, 8 GB reference machine.
- **Statement:** The team makes **zero claims** regarding the organizer's private test set results. All algorithmic mechanisms (FFT-NCC, candidate clustering, subpixel paraboloid fit) are fully deterministic and generalizable without test-set memorization.

### B. Released Development-Set Measurement
- **Score:** **90.50 / 100.00**
- **Dataset:** The 180 development pairs released by Applied Materials (`data/phase2_dev/pairs.csv`: 70 Set A, 70 Set B, 40 Set C).
- **Rubric:** Exactly follows `phase2/V27_FINAL/SCORER_AUDIT.md`:
  - Localization (40%): **40.00 / 40** (100% $\le 5\text{ px}$)
  - Pose Recovery (20%): **19.20 / 20** (Rotation MAE $\le 0.065^\circ$, Scale MAE $\le 0.056$)
  - Absence Rejection (15%): **8.09 / 15** (Set C F1: 0.539, 38 TN, 2 FP)
  - Confidence Calibration (10%): **8.27 / 10** (Spearman $\rho = 0.832$)
  - Runtime Efficiency (5%): **5.00 / 5** (0.07 s/pair median $\ll 5.0\text{ s}$)
  - Documentation & Compliance (10%): **10.00 / 10** (Strict 7-column schema)

### C. Historical Diagnostic Scores
- Throughout Phase 2 R&D, internal diagnostic scripts produced varying intermediate numbers (e.g. V25 at 87.02, V28 at 88.39, V46 unconstrained candidate recall).
- **Statement:** These numbers were internal exploratory checkpoints. Only the frozen **90.50** pipeline shipped in `FINAL_SUBMISSION/` is authoritative.

### D. Multimodal RGB Bonus Channel
- The team developed an end-to-end RGB localization branch (`rgb_branch.py` using Rec. 601 luminance and dual-channel gradient matching), achieving **0.00 px error** on the provided `rgb_bonus_package`.
- **Statement:** This capability is fully implemented and tested, but bonus points depend entirely on organizer evaluation of Set D and are **not** factored into the 90.50 score.
