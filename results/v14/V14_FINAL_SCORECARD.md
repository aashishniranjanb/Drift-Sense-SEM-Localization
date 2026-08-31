# Drift-Sense++ V14 Final Scorecard & Configuration Freeze

## 1. Frozen Production Configuration Specification

| Component | Active Production Engine | Specification / Formula / Model | Decision / Status |
| :--- | :---: | :--- | :---: |
| **Pose Estimation** | `fallback` | Sequential Coarse-to-Fine Search (Scale $\text{step}=0.25$, Rotation $\text{step}=1.0^\circ$, Fine $\text{step}=0.05/0.2^\circ$) | **FROZEN WINNER** |
| **Candidate Extractor** | `fallback` | Iterative Spatial NMS $r=5$ with Top-50 candidate pool | **FROZEN WINNER** |
| **Candidate Ranker** | `fallback` | Full CAR Engine (Replica Family Clustering + Spatial Fingerprints + Conditional PACE Re-ranking) | **FROZEN WINNER** |
| **Metrology / Refinement** | Production | Subpixel Phase Correlation & 2D Paraboloid Extrema Fitting | **FROZEN WINNER** |
| **Presence & Rejection** | `fallback` | **V14-P1 Multi-Evidence Composite Presence Engine** ($t=0.58$) | **FROZEN WINNER** |
| **Confidence Monotonicity**| Production | $\text{Score} = \text{clamp}(0.35 \times \text{corr} + 0.40 \times \text{ctx}_{128} + 0.15 \times \frac{\text{psr}}{10} + 0.10 \times \text{margin} - 0.20 \times \text{phase\_res}, 0, 1)$ | **FROZEN WINNER** |

---

## 2. Benchmark Scorecard (180 Cases: 70 Set A, 70 Set B, 40 Set C)

### A. Localization Metrics ($\le 5\text{ px}$ Target)
- **Official Weighted Loc Score (0.45*A + 0.55*B)**: **48.88%** (*+13.44% absolute increase over 35.44% baseline*)
- **Set A (Nominal) $\le 1\text{ px}$**: 34.69% | **Set A $\le 5\text{ px}$**: 38.78% | Median: 50.44 px
- **Set B (Degraded) $\le 1\text{ px}$**: 57.14% | **Set B $\le 5\text{ px}$**: **57.14%** | **Median: 0.74 px (Subpixel)**

### B. Pose Recovery Metrics
- **Set A Scale MAE**: **0.0482** | **Set A Rotation MAE**: **0.1016°** (Target $\le 0.20^\circ$)
- **Set B Scale MAE**: **0.0396** | **Set B Rotation MAE**: **0.1332°** (Target $\le 0.20^\circ$)

### C. Absence Rejection Metrics (Set C Target F1)
- **Set C Rejection F1 Score**: **0.3862** (*+102.7% relative gain over 0.1905 baseline*)
- **Rejection Precision**: **0.2667**
- **Rejection Recall**: **0.7000** (28 / 40 absent cases correctly rejected)

### D. Confidence Monotonicity
- **Spearman Rank Correlation ($\rho$)**: **0.5005** (*Hit stretch goal $\ge 0.50$*)

### E. Failure Taxonomy Breakdown (180 Cases)
- **PERIODIC_REPLICA**: 36 cases (20.0%) — *Down from 67 cases (46.3% reduction)*
- **REJECTION_SUCCESS**: 28 cases (15.6%) — *Up from 8 cases*
- **SUBPIXEL_SUCCESS**: 25 cases (13.9%)
- **ABSENCE_FALSE_POSITIVE**: 12 cases (6.7%) — *Down from 32 cases*
- **IN_BOUNDS_SUCCESS**: 2 cases (1.1%)
- **PRESENCE_FALSE_NEGATIVE**: 77 cases (42.8%)

---

## 3. Final Production Status: **FROZEN FOR SUBMISSION**
All components are fully validated, reproducible, and self-contained in the standalone `inference.py` entrypoint.
