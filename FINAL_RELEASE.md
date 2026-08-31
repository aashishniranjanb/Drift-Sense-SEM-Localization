# Drift-Sense++ Phase 2 Final Release

## 1. Release Overview
- **Release Tag**: `V14-FINAL`
- **Release Commit**: `37b1895`
- **Official Evaluation Dataset**: `data/phase2_dev/pairs.csv` (180 cases: 70 Set A, 70 Set B, 40 Set C)
- **Status**: **FROZEN FOR SUBMISSION**

---

## 2. Production Architecture & Configuration

| Subsystem | Production Module / Engine | Exact Parameters / Formulation |
| :--- | :---: | :--- |
| **Pose Estimation** | `fallbacks/pose_fallback.py` | Decoupled sequential coarse-to-fine scale sweep ($s \in [8, 12]$, step $0.25 \to 0.05$) and rotation sweep ($\theta \in [-5, +5]^\circ$, step $1.0^\circ \to 0.2^\circ$) |
| **Candidate Retrieval** | `fallbacks/ranking_fallback.py` | Iterative Spatial NMS with optimal suppression radius $r=5$ px, extracting Top-50 candidate pool |
| **Candidate Ranking** | `fallbacks/ranking_fallback.py` | Confidence-Adaptive Ranking (CAR) with replica family clustering, spatial fingerprinting, and conditional PACE context re-ranking |
| **Subpixel Metrology** | `phase2/pose_refinement.py` | Dual subpixel metrology using phase correlation refinement and 2D paraboloid extrema fitting |
| **Presence & Rejection** | `fallbacks/rejection_fallback.py` | **V14-P1 Multi-Evidence Composite Presence Engine** ($t=0.58$ threshold):<br>$$\text{Score} = \text{clamp}(0.35 \times \text{corr} + 0.40 \times \text{ctx}_{128} + 0.15 \times \frac{\text{psr}}{10} + 0.10 \times \text{margin} - 0.20 \times \text{phase\_residual}, 0, 1)$$ |
| **Component Selector** | `production_engine/config.py` | `POSE_ENGINE = "fallback"`, `RANKING_ENGINE = "fallback"`, `REJECTION_ENGINE = "fallback"` |

---

## 3. Verified Benchmark Scorecard (180 Cases)

### A. Localization Metrics ($\le 5\text{ px}$ Target)
- **Official Weighted Localization Score (0.45*A + 0.55*B)**: **48.88%** (*+13.44% absolute improvement over 35.44% baseline*)
- **Set A (Nominal) $\le 1\text{ px}$**: 34.69% | **Set A $\le 5\text{ px}$**: 38.78% | Median Error: 50.44 px
- **Set B (Degraded) $\le 1\text{ px}$**: 57.14% | **Set B $\le 5\text{ px}$**: **57.14%** | **Median Error: 0.74 px (Subpixel Accuracy)**

### B. Pose Recovery Metrics
- **Set A Scale MAE**: **0.0482** | **Set A Rotation MAE**: **0.1016°** (Target $\le 0.20^\circ$)
- **Set B Scale MAE**: **0.0396** | **Set B Rotation MAE**: **0.1332°** (Target $\le 0.20^\circ$)

### C. Absence Rejection Metrics (Set C Target F1)
- **Set C Rejection F1 Score**: **0.3862** (*+102.7% relative gain over 0.1905 baseline*)
- **Rejection Precision**: **0.2667**
- **Rejection Recall**: **0.7000** (28 / 40 absent cases correctly rejected)

### D. Confidence Monotonicity
- **Spearman Rank Correlation ($\rho$)**: **0.5005** (*Hit stretch target $\ge 0.50$*)
- **High-Confidence Accuracy**: **83.1% to 100.0%** decision accuracy in confidence bins $[0.6, 1.0]$.

### E. Failure Taxonomy Breakdown (180 Cases)
- **PERIODIC_REPLICA**: 36 cases (20.0%) — *Reduced by 46.3% from 67 baseline cases*
- **REJECTION_SUCCESS**: 28 cases (15.6%) — *Up from 8 cases*
- **SUBPIXEL_SUCCESS**: 25 cases (13.9%)
- **ABSENCE_FALSE_POSITIVE**: 12 cases (6.7%) — *Down from 32 cases*
- **IN_BOUNDS_SUCCESS**: 2 cases (1.1%)
- **PRESENCE_FALSE_NEGATIVE**: 77 cases (42.8%)

---

## 4. Documented Negative Results & Rejected Formulations

| Experiment | Formulation | Result | Decision |
| :--- | :--- | :---: | :---: |
| **V14-R2 Ranker** | Unconditional linear context-128 / phase weighting | Weighted Loc dropped from **48.88% $\to$ 8.43%** | **REJECTED** |
| **V13-Exp1 Quotas** | Fixed local grid quota candidate extraction | Top-100 Recall dropped from **60.71% $\to$ 54.29%** | **REJECTED** |
| **V13-Exp3 Z-Score** | Local correlation Z-score hypothesis fusion | Top-100 Recall dropped from **59.29% $\to$ 56.43%** | **REJECTED** |

---

## 5. Standalone Competition Execution Interface

The primary competition entrypoint is strictly self-contained and requires no external network access:

```bash
# Basic inference: returns predicted coordinates (x.xx, y.yy) or rejection
python inference.py --reference data/phase2_dev/reference/pair_001.png --search data/phase2_dev/search/pair_001.png

# Verbose inference: returns structured JSON with pose, presence, and confidence metadata
python inference.py --reference data/phase2_dev/reference/pair_001.png --search data/phase2_dev/search/pair_001.png --verbose
```

---

## 6. Submission Verification Checklist

- [x] V14 benchmark reproducible on dev dataset (`data/phase2_dev/pairs.csv`)
- [x] Official weighted localization score: **48.88%** verified
- [x] V14-P1 presence engine active ($t=0.58$) in `fallbacks/rejection_fallback.py`
- [x] Production config gated via `production_engine/config.py`
- [x] Standalone `inference.py` CLI tested on Set A (nominal), Set B (degraded), and Set C (absent)
- [x] Subpixel accuracy on degraded Set B verified (Median: **0.74 px**)
- [x] No external network access or hardcoded machine-specific absolute paths
- [x] Requirements file complete (`requirements.txt`)
- [x] Release tag `V14-FINAL` created
