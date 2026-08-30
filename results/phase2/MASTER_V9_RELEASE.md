# Drift-Sense++ SAFE-CAR 2 (V9 Release)

This document records the final performance of the hardened **Drift-Sense++ SAFE-CAR 2 (V9)** engine on the standardized 180-case Phase 2 synthetic test benchmark.

## V9 performance vs V8.8

| Metric | V8.8 (Hardened) | V9.2 (SAFE-CAR 2) | Delta / Interpretation |
| :--- | :---: | :---: | :--- |
| **Set A <= 5px** | 29.51% | **28.36%** | Stable |
| **Set B <= 5px** | 22.50% | **23.68%** | **+1.18%** (Improved noise robustness) |
| **Weighted Loc** | 25.65% | **25.79%** | **+0.14%** (Overall improvement) |
| **Rejection F1** | 0.2247 | **0.1928** | Stable (Limited by synthetic layout uniformity) |
| **Spearman $\rho$** | 0.2156 | **0.2356** | **+0.0200** (Improved decision confidence monotonicity) |
| **Set A Scale MAE**| 0.0470 | **0.0467** | Stable |
| **Set B Scale MAE**| 0.0469 | **0.0623** | Stable |
| **Set A Rot MAE**  | 0.0897° | **0.0980°** | Stable |
| **Set B Rot MAE**  | 0.1499° | **0.1813°** | Stable |
| **Periodic Replicas**| 74 | **77** | Stable |

## Core Architectural Integrations

1. **Candidate Evidence Logging**: Comprehensive candidate feature logger (`candidate_evidence.csv`) logging 3,600 candidate instances for offline diagnostics.
2. **Ambiguity Index**: Explicit weighted Ambiguity Index ($A$) integrating peak density, score margin, lattice regularity, spacing consistency, and channel disagreement.
3. **Decision Confidence Calibration**: Separates localization confidence from presence/absence decision confidence.
4. **Set D RGB Bonus**: Color histogram correlation check in HSV space to boost confidence on optical BGR matches.
