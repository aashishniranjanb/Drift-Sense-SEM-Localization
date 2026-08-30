# V8.0 Preliminary 20-Case Baseline Report

## Summary Metrics
- **Dataset**: 20 preliminary synthetic cases (19 Present, 1 Cross-Architecture Absent)
- **Presence Rejection F1**: `0.9730` (Precision: 1.0000, Recall: 0.9474)
- **Scale Recovery MAE**: `0.0306`
- **Rotation Recovery MAE**: `0.0629°`
- **Localization ($\le 5\text{px}$)**: `33.33%`
- **Median Localization Error**: `202.08 px`
- **Spearman Monotonicity ($\rho$)**: `0.3909`

## Identified Baseline Weaknesses
1. **Unrealistic Set C**: Absent case tested DRAM template against FinFET search image, inflating rejection score.
2. **Periodic Replica Misalignment**: Median error of 202 px indicates scale and rotation are accurate, but spatial candidate selection selects replica cells.
3. **Weak Calibration**: Manual weighted score sum yielded low Spearman rank correlation ($\rho = 0.3909$).
4. **Lack of A/B Decoupling**: Dataset did not evaluate Set A (nominal) and Set B (degraded) independently.
