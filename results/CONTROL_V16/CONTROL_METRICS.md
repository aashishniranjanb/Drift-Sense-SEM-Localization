# V16 Frozen Control Baseline Specification

## 1. Frozen System Identification
- **Baseline ID**: `V16_CONTROL_FROZEN`
- **Commit**: see `CONTROL_COMMIT.txt`
- **Date**: 2026-09-01
- **Status**: **IMMUTABLE REFERENCE CONTROL**

---

## 2. Official Competition 100-Point Scorecard Breakdown

| Evaluation Dimension | Weight | V16 Control Raw Metric | Points Awarded | Target Points |
| :--- | :---: | :---: | :---: | :---: |
| **Localization** | **40 pts** | **50.81%** (Weighted Loc: $0.45 \times A + 0.55 \times B$) | **20.32 / 40** | **34 - 38 / 40** |
| **Pose Recovery** | **20 pts** | Scale MAE: 0.0485 / Rot MAE: 0.1170° | **16.50 / 20** | **18 - 20 / 20** |
| **Absence Rejection** | **15 pts** | F1: **0.3862** (Precision: 0.2667, Recall: 0.7000) | **5.79 / 15** | **13 - 15 / 15** |
| **Confidence Calibration** | **10 pts** | Spearman $\rho$: **0.1270** / AUC: ~0.62 | **4.00 / 10** | **9 - 10 / 10** |
| **Efficiency & Latency** | **5 pts** | Median Latency: **~4.8s** / Mean: **5.8s** | **4.50 / 5** | **5.00 / 5** |
| **Generator & Analysis** | **10 pts** | Standalone deterministic generator + citations | **10.00 / 10** | **10.00 / 10** |
| **TOTAL ESTIMATED SCORE** | **100 pts** | — | **61.11 / 100** | **89 - 95 / 100** |

---

## 3. Granular Error Distribution (180 Grayscale Cases)
- **Set A (Nominal 70 cases)**:
  - $\le 1\text{ px}$: 35.42%
  - $\le 5\text{ px}$: 39.58%
  - Median error: 42.03 px
- **Set B (Degraded 70 cases)**:
  - $\le 1\text{ px}$: 60.00%
  - $\le 5\text{ px}$: 60.00%
  - Median error: 0.70 px (Subpixel)
- **Set C (Absent 40 cases)**:
  - Correctly rejected: 28 / 40 (70.0% Recall)
  - False positives: 12 / 40
  - False negatives on Present (Set A+B): 77 / 140 (55.0%)

---

## 4. Failure Taxonomy Summary
- **PRESENCE_FALSE_NEGATIVE**: 77 cases (42.8%)
- **PERIODIC_REPLICA**: 35 cases (19.4%)
- **REJECTION_SUCCESS**: 28 cases (15.6%)
- **SUBPIXEL_SUCCESS**: 26 cases (14.4%)
- **ABSENCE_FALSE_POSITIVE**: 12 cases (6.7%)
- **IN_BOUNDS_SUCCESS**: 2 cases (1.1%)
