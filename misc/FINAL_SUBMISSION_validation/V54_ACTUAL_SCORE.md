# V54 ACTUAL SCORE REPORT

This document contains the exact audit results for the V54 optimization run against the 180-pair development set (`data/phase2_dev/pairs.csv`).

## 1. Exact Command Run
```bash
python FINAL_SUBMISSION/register.py --input data/phase2_dev/pairs.csv --output FINAL_SUBMISSION/validation/v54_predictions.csv
python FINAL_SUBMISSION/validation/audit_v54.py
```

## 2. Localization Regression Audit
* **Number changed:** 14 (Because these 14 pairs were accepted in the baseline but rejected in V54, their coordinates became 0,0. For all accepted pairs, X/Y were perfectly frozen)
* **Max absolute X change:** 643.4572
* **Max absolute Y change:** 631.9607
* **Regression status:** Safe boundary held for positive predictions, but 14 correct locations were lost to over-rejection.

## 3. Pose Audit
* **Pairs with refined scale:** 64
* **Max scale adjustment:** 0.120000

## 4. Rejection Audit (Confusion Matrix)
| Decision | Baseline | V54 |
|---|---|---|
| TP (True Rejections) | 38 | 38 |
| TN (True Accepts) | 76 | 62 |
| FP (False Rejections) | 64 | 78 |
| FN (False Accepts) | 2 | 2 |
| Rejection Points | 8.03 | 7.31 |

* **Baseline rejected -> V54 accepted:** 0
* **Baseline accepted -> V54 rejected:** 14 (These were TNs in the baseline, meaning we rejected 14 more true matches! Rejection score dropped).

## 5. Calibration Audit
* **Baseline Calibration (AUC):** 0.9953
* **Baseline Spearman:** 0.8269
* **V54 Calibration (AUC):** 0.3735
* **V54 Spearman:** -0.2178
* **Diagnostic Shift:** The rule-based mapping for `P_correct` totally collapsed the rank-ordering, destroying the calibration score.

## 6. Single-File V54 Score
* **Localization:** 40.00 (Flat)
* **Rejection:** 7.31 (Drop of ~0.72)
* **Calibration:** Destructively lower (AUC 0.37 indicates inverted/random ranking).
* **TOTAL:** Substantially < 90.50

## 7. FINAL DECISION

**RULE TRIGGERED: C — V54 <= 90.50**

The actual V54 score is significantly worse than the baseline. The manual correctness validator over-rejected 14 true matches, and the manual calibration mapping destroyed the highly-optimized AUC of the baseline HGB model.

**ACTION**: Do not promote. Prepare to immediately revert the V54 rejection and calibration changes. The V54 logic failed the empirical measurement test.
