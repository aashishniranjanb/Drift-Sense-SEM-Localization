# Master V8 Ablation Study Report

This document records the full comparative evaluation of the **Drift-Sense++ V8** development lineage (Phases 3 to 9) on the standardized 180-case Phase 2 synthetic test benchmark (consisting of 70 Set A Nominal, 70 Set B Degraded, and 40 Set C Absent pairs).

## Ablation Metrics Summary Table

| Phase / Version | Set A $\le$ 5px (%) | Set B $\le$ 5px (%) | Weighted Loc (%) | Rejection F1 | Spearman $\rho$ | Scale MAE (A / B) | Rotation MAE (A / B) | Periodic Replicas | Presence False Negatives | Absence False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V8.1 (Baseline)** | 26.15% | 14.58% | 19.79% | 0.1644 | 0.2906 | 0.0484 / 0.0482 | 0.0868° / 0.1520° | 89 | 27 | 34 |
| **V8.2 (Context Matcher)** | 19.67% | 19.05% | 19.33% | 0.2093 | 0.2370 | 0.0488 / 0.0480 | 0.0896° / 0.1305° | 83 | 37 | 31 |
| **V8.3 (Consensus - Rejected)**| 18.75% | 12.20% | 15.14% | 0.2558 | 0.1894 | 0.0489 / 0.0477 | 0.0874° / 0.1384° | 88 | 35 | 29 |
| **V8.4 (Phase Consistency)**| 19.67% | 19.51% | 19.58% | 0.2273 | 0.2034 | 0.0473 / 0.0480 | 0.0890° / 0.1298° | 82 | 38 | 30 |
| **V8.5 (Lattice Detector)** | 29.51% | 22.50% | 25.65% | 0.2247 | 0.2261 | 0.0470 / 0.0469 | 0.0897° / 0.1499° | 74 | 39 | 30 |
| **V8.6 (PACE Reranking)** | 29.51% | 22.50% | 25.65% | 0.2247 | 0.2261 | 0.0470 / 0.0469 | 0.0897° / 0.1499° | 74 | 39 | 30 |
| **V8.7 (Calibration)** | 29.51% | 22.50% | 25.65% | 0.2247 | 0.2156 | 0.0470 / 0.0469 | 0.0897° / 0.1499° | 74 | 39 | 30 |
| **V8.8 (Hardening)** | 29.51% | 22.50% | 25.65% | 0.2247 | 0.2156 | 0.0470 / 0.0469 | 0.0897° / 0.1499° | 74 | 39 | 30 |

## Strategic Interpretations

1. **Transformative Pose Stability**: Scale MAE (under 0.05) and Rotation MAE (under 0.15°) are highly stable across nominal and heavily degraded datasets, confirming that transformation estimation is solved.
2. **Periodic Replica Mismatches**: This is the single largest bottleneck, accounting for 74-89 failure cases (over 40% of the entire dataset). Classical cross-correlation without neighborhood checks fails to identify the correct physical target cell.
3. **Rejection Insufficiency**: Set C Rejection F1 is capped under 0.26. Fixed correlation thresholds are inadequate to handle low e-beam dose images (Set B) where noise suppresses correlation peaks.
4. **Ranking Monotonicity**: Spearman correlation is severely degraded by overconfident False Negatives (present cases rejected as absent with high confidence).
