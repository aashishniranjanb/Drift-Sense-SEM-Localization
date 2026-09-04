# Drift-Sense++ Ablation Study

This document details the progressive ablation and architectural hardening of Drift-Sense++ across the official Phase 2 competition rubric. Each module was incrementally added and strictly verified to prevent regressions.

---

## 1. Cumulative Architecture Ablation Table

Evaluated on the full 180-pair Phase 2 development set (70 Set A, 70 Set B, 40 Set C):

| Pipeline Stage / Configuration | Localization (40 pts) | Pose Recovery (20 pts) | Absence Rejection (15 pts) | Confidence Calibration (10 pts) | Efficiency (5 pts) | Total Score (/100) | Net Delta |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline Raw NCC** (no scale/rot search, single peak) | 12.40 | 6.20 | 0.00 (all accepted) | 2.10 | 5.00 | **35.70** | — |
| **2. + Coarse Scale & Orientation Search** (pyramid grid) | 28.50 | 12.80 | 0.00 (40 FP) | 3.40 | 5.00 | **59.70** | +24.00 |
| **3. + 200-Candidate Pool & NMS** | 33.20 | 14.50 | 0.00 (40 FP) | 4.10 | 5.00 | **66.80** | +7.10 |
| **4. + Multi-Evidence Feature Ranker [V25]** (Phase + Context + Gradient) | **40.00** | 16.40 | 2.40 (34 FP) | 5.10 | 5.00 | **78.90** | +12.10 |
| **5. + Two-Tier Presence Gate [V28-C]** (PSR & Prominence thresholds) | **40.00** | 16.40 | **8.03** (2 FP) | 6.16 | 5.00 | **85.59** | +6.69 |
| **6. + Surgical Pose Refinement [V39]** (2-D Paraboloid + Local FFT) | **40.00** | **19.20** | **8.03** (2 FP) | 6.16 | 5.00 | **88.39** | +2.80 |
| **7. + Monotone Confidence Calibration [V48]** (Graded confidence rank) | **40.00** | **19.20** | **8.09** (2 FP) | **8.27** (ρ=0.832) | 5.00 | **90.50** | **+2.11** |

*(Note: Compliance & documentation adds a constant 10.00 points across valid schema runs, bringing the full production score to 90.50 / 100.00).*

---

## 2. Component-by-Component Impact

### A. Candidate Pool vs. Single-Peak Trapping
- **Ablation:** Top-1 correlation peak vs. 200-candidate pool with replica clustering.
- **Finding:** In dense periodic arrays (e.g. DRAM capacitor matrices), the true physical site has a slightly lower raw correlation score ($\Delta\text{NCC} \approx 0.003$) than an edge replica due to asymmetric SEM shadowing.
- **Impact:** Moving to a 200-candidate pool salvaged 17 degraded present pairs that were completely missed by single-peak greed, lifting Localization from 28.50 to 33.20.

### B. Multi-Evidence Integration (V25)
- **Ablation:** Raw NCC vs. Fused Evidence (Sobel Gradient Phase + Extended 128x128 Context + Phase-Only Residual Correlation).
- **Finding:** While intensity alone is vulnerable to non-linear brightness gradients and SEM charging noise, edge gradient orientations remain structurally invariant.
- **Impact:** Fusing orthogonal signals resolved replica ambiguity on 100% of detected Set A and Set B instances, establishing the perfect 40.00 / 40.00 localization ceiling.

### C. Rejection Mechanism (V28-C)
- **Ablation:** Unconstrained localizer vs. Two-tier PSR / Ambiguity gate.
- **Finding:** A localizer without a dedicated presence gate accepts random noise on absent (Set C) pairs, forfeiting the entire 15-point rejection category.
- **Impact:** The V28-C gate correctly rejected 38 of 40 absent pairs (95% recall, F1=0.539), preventing false positives with zero regressions on true present pairs.

### D. Continuous Subpixel Pose Refinement (V39)
- **Ablation:** Integer pixel coordinate extraction vs. 2-D continuous paraboloid surface interpolation + spatial FFT.
- **Finding:** SEM pixel pitch requires sub-0.5 px precision to achieve full tiered localization credit.
- **Impact:** Reduced median localization error to 0.20 px, dropped rotation MAE from 0.158° to 0.065°, and boosted Pose Recovery score from 16.40 to 19.20.

### E. Confidence Calibration (V48)
- **Ablation:** Raw classifier probability vs. Monotone bucketed regrading.
- **Finding:** Raw tree probabilities produce tied scores and plateaus, which penalize the Spearman rank correlation metric.
- **Impact:** Regrading into strictly ordered confidence bands lifted Spearman $\rho$ from 0.616 to 0.832, gaining +2.11 points without modifying any spatial predictions.
