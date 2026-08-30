# V10 Final Freeze Metrics

This document archives the frozen performance of the **V10.0 (SAFE-CAR 2)** baseline before the V11 candidate-recovery modifications.

## 1. Dev Benchmark Summary (180 Cases)

- **Set A <= 5px**: 25.00%
- **Set B <= 5px**: 27.03%
- **Weighted Localization**: 26.11%
- **Rejection F1**: 0.1928
- **Spearman rho**: 0.2554
- **Scale MAE**: 0.0466 (Set A) / 0.0639 (Set B)
- **Rotation MAE**: 0.0989° (Set A) / 0.1813° (Set B)
- **Average Latency**: 2.55 seconds

## 2. Adversarial Benchmark Summary (150 Cases)

- **01_exact_periodic_replica**: 40.0%
- **02_near_periodic_replica**: 53.3%
- **03_phase_shifted_replica**: 40.0%
- **04_noise_degraded**: 33.3%
- **05_charging_degraded**: 40.0%
- **06_scale_extreme**: 20.0%
- **07_rotation_extreme**: 20.0%
- **08_absent_same_architecture**: 0.0% (Rejection F1 = 0.00)
- **09_absent_different_architecture**: 0.0% (Rejection F1 = 1.00)
- **10_combined_failure**: 6.7%
