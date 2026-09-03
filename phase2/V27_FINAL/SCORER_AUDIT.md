# PHASE 2 SCORER AUDIT: EXACT RULES & FORMULAS

Source: phase2/benchmark_phase2.py

## 1. Localization Metric (Weight: 40 points)
* Evaluated strictly on **present ground-truth pairs** (gt_found == 1) that are **accepted by the prediction** (pred_found == 1).
* If a pair is rejected (pred_found == 0), it is **excluded from the localization error array**.
* Accuracy Tiers:
  - le_1: mean(loc_err <= 1.0) * 100%
  - le_5: mean(loc_err <= 5.0) * 100%
  - Median error in pixels.
* **Official Weighted Localization Score**:
  \text{Weighted Loc Score} = 0.45 \times \text{SetA}_{le5} + 0.55 \times \text{SetB}_{le5}
* Points awarded:
  \text{Localization Points} = \text{Weighted Loc Score} \times 0.40

## 2. Pose Recovery Metrics (Weight: ~18-20 points)
* Evaluated only on accepted present pairs (gt_found == 1 and pred_found == 1).
* Computes Mean Absolute Error (MAE):
  - Scale MAE: mean(abs(scale - gt_scale))
  - Rotation MAE: mean(abs(theta - gt_theta))
* Set A and Set B reported independently.

## 3. Absence Rejection Metrics (Weight: 15 points)
* Evaluated across all 180 pairs where **absence (ound == 0) is the positive class**.
  - TP: gt_found == 0 and pred_found == 0 (True Rejection)
  - FP: gt_found == 1 and pred_found == 0 (False Rejection / Presence False Negative)
  - FN: gt_found == 0 and pred_found == 1 (False Accept / Absence False Positive)
  - TN: gt_found == 1 and pred_found == 1 (True Accept / Present Detected)
* Metrics:
  \text{Precision} = \frac{TP}{TP + FP}
  \text{Recall} = \frac{TP}{TP + FN}
  \text{Rejection F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
* Points awarded:
  \text{Rejection Points} = \text{Rejection F1} \times 15.0

## 4. Confidence Monotonicity / Calibration (Weight: 10 points)
* Evaluated across all 180 pairs.
* Correctness label  \in \{0, 1\}$ defined as:
  - $ if failure mode is SUBPIXEL_SUCCESS (\_err \le 1.0$)
  - $ if failure mode is IN_BOUNDS_SUCCESS (\_err \le 5.0$)
  - $ if failure mode is REJECTION_SUCCESS (=0, pred=0$)
  - $ otherwise (PERIODIC_REPLICA, PRESENCE_FALSE_NEGATIVE, ABSENCE_FALSE_POSITIVE).
* **Official Metric**: Spearman Rank Correlation ($\rho$):
  \rho = \text{spearmanr}(\text{score}, y)
* Points awarded:
  \text{Calibration Points} = \rho \times 10.0

## 5. Efficiency & Compliance (Weight: 15 points)
* Runtime constraint: Median runtime $\le 5.0$ seconds per pair.
* Schema compliance: pair_id, x, y, theta, scale, found, score.
* Zero-coordinate rule: If ound == 0, coordinates and pose fields must be zero: x=0.0, y=0.0, theta=0.0, scale=0.0.
