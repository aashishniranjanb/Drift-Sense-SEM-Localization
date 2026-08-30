# Baseline Phase 2 Aashish 001

This document freezes the authoritative control baseline metrics for the Aashish Fallback Stack.

## 1. Environment & Setup Metadata
- **Baseline Identifier**: `BASELINE_PHASE2_AASHISH_001`
- **Git Commit**: `c3484ce`
- **Dataset**: `data/phase2_dev/pairs.csv`
- **Row Count**: 180 total pairs (70 Set A, 70 Set B, 40 Set C)
- **Active Code Base**: `production_engine/production_runner.py` with `config.py` component selectors set to `"fallback"`.

---

## 2. Quantitative Control Metrics

### A. Localization Metrics (Target $\le 5\text{ px}$)
- **Weighted Localization Score (0.45*A + 0.55*B)**: **35.44%**
- **Set A $\le 1\text{ px}$**: 32.84% | **Set A $\le 5\text{ px}$**: 35.82% | **Set A Median**: 50.44 px
- **Set B $\le 1\text{ px}$**: 32.43% | **Set B $\le 5\text{ px}$**: 35.14% | **Set B Median**: 75.15 px

### B. Pose Recovery Metrics
- **Set A Scale MAE**: 0.0467 | **Set A Rotation MAE**: 0.0980°
- **Set B Scale MAE**: 0.0639 | **Set B Rotation MAE**: 0.1813°

### C. Absence Rejection Metrics (Target F1 > 0.90)
- **Set C Rejection F1 Score**: **0.1905**
- **Rejection Precision**: 0.1818 | **Rejection Recall**: 0.2000

### D. Confidence Monotonicity
- **Spearman Rank Correlation (rho)**: 0.1396

---

## 3. Failure Taxonomy Breakdown
- **PERIODIC_REPLICA**: 67 cases (37.2%)
- **PRESENCE_FALSE_NEGATIVE**: 36 cases (20.0%)
- **SUBPIXEL_SUCCESS**: 34 cases (18.9%)
- **ABSENCE_FALSE_POSITIVE**: 32 cases (17.8%)
- **REJECTION_SUCCESS**: 8 cases (4.4%)
- **IN_BOUNDS_SUCCESS**: 3 cases (1.7%)
