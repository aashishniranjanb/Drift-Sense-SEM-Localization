# Baseline Confusion & Decision Matrix Analysis

## 1. Metric Definition Resolution

The competition scoring evaluates **Absence Rejection** (where `found == 0` is the target positive class for Set C rejection):

### A. Absence Rejection Perspective (Official Set C Benchmark Metric: Target = Found 0)
*   **True Positive (Absence Detected correctly)**: 8 cases
*   **False Positive (Present case incorrectly rejected)**: 36 cases (`PRESENCE_FALSE_NEGATIVE` = 36)
*   **False Negative (Absent case falsely accepted)**: 32 cases (`ABSENCE_FALSE_POSITIVE` = 32)
*   **True Negative (Present case correctly accepted)**: 104 cases
*   **Rejection Precision**: 0.1818 (8/44)
*   **Rejection Recall**: 0.2000 (8/40)
*   **Rejection F1 Score**: **0.1905**

### B. Presence Detection Perspective (Target = Found 1)
*   **True Positive (Present localized)**: 104 cases
*   **False Positive (Absent localized)**: 32 cases
*   **False Negative (Present rejected)**: 36 cases
*   **True Negative (Absent rejected)**: 8 cases
*   **Presence Precision**: 0.7647
*   **Presence Recall**: 0.7429
*   **Presence F1 Score**: 0.7536

---

## 2. Failure Taxonomy Summary (180 cases)
*   **PERIODIC_REPLICA**: 67 cases (37.2%)
*   **PRESENCE_FALSE_NEGATIVE**: 36 cases (20.0%)
*   **ABSENCE_FALSE_POSITIVE**: 32 cases (17.8%)
*   **SUBPIXEL_SUCCESS**: 34 cases (18.9%)
*   **REJECTION_SUCCESS**: 8 cases (4.4%)
*   **IN_BOUNDS_SUCCESS**: 3 cases (1.7%)

All 180 individual predictions with ground truth and error records are saved in `results/v14/baseline_confusion.csv`.
