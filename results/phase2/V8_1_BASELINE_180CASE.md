# V8.1 Hardened 180-Case Baseline Report

## Summary Metrics
- **Dataset**: 180 standardized synthetic cases (70 Set A Nominal, 70 Set B Degraded, 40 Set C Same-Architecture Hard Negatives)
- **Official Weighted Localization Score**: `14.39%` (Set A <= 5 px: 26.15%, Set B <= 5 px: 4.76%)
- **Set C Rejection F1 Score**: `0.3063` (Precision: 0.2394, Recall: 0.4250)
- **Pose Recovery MAE**:
  - Set A: Scale MAE = `0.0484` | Rotation MAE = `0.0868°`
  - Set B: Scale MAE = `0.0814` | Rotation MAE = `0.2557°`
- **Spearman Monotonicity ($\rho$)**: `0.1171`

## Failure Taxonomy Breakdown
- **PERIODIC_REPLICA**: 68 cases (37.8%) -> Matches incorrect periodic grid cells.
- **PRESENCE_FALSE_NEGATIVE**: 54 cases (30.0%) -> Present targets falsely classified as absent.
- **ABSENCE_FALSE_POSITIVE**: 23 cases (12.8%) -> Absent targets falsely classified as present.
- **SUBPIXEL_SUCCESS**: 17 cases (9.4%) -> Accurate subpixel matches <= 1 px.
- **REJECTION_SUCCESS**: 17 cases (9.4%) -> Correctly rejected absent cases.
- **IN_BOUNDS_SUCCESS**: 1 case (0.6%) -> Accurate matches between 1 px and 5 px.

---

## Action Plan for Subsequent Phases
1. **Resolve Periodic Replicas (Phase 3 & 6)**: Implement multi-scale context descriptors ($32^2, 64^2, 128^2$) and a periodic lattice grid detector to distinguish identical periodic structures from the correct physical neighborhood.
2. **Improve Set B Robustness (Phase 4 & 5)**: The low localization accuracy (4.76%) on degraded images requires integrating dual-channel consensus and phase consistency checks.
3. **Calibrate Rejection and Monotonicity (Phase 8)**: Standardize evidence inputs to improve the low Spearman correlation ($\rho = 0.1171$).
