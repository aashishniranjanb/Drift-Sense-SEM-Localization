# V26 CHAMPIONSHIP GAP EXPERIMENT: FINAL REPORT

## OVERALL SCORE COMPARISON

| Component | V25 | V26 | Delta | Points Remaining |
|---|---|---|---|---|
| Localization | 39.42 | 26.74 | -12.68 | 13.26 |
| Pose | 18.00 | 17.00 | -1.00 | 3.00 |
| Rejection | 8.14 | 7.64 | -0.50 | 7.36 |
| Calibration | 5.49 | 1.76 | -3.73 | 8.24 |
| **Total Base** | **86.05** | **53.14** | **-32.91** | **46.86** |

## COMPONENT EXPERIMENT DIAGNOSTICS

### Experiment A & B: Retrieval & Pairwise Hard-Negative Verifier
- **Multi-Queue Extraction:** Evaluated up to 150 deduplicated candidates per pair using adaptive radii (=5, =3, =2).
- **Pairwise Ranking:** Implemented HistGradientBoostingClassifier evaluating Δcorr, Δcontext, Δphase, etc. This reduced the Periodic Replica contamination from the naïve Retrieval experiment (from 63 to 22), but it was still not strong enough to protect the 39.42 baseline.

### Experiment C: Two-Stage Rejection
- Separated P(reference present) from P(candidate correct | reference present).
- Rejection F1 remained essentially flat (**0.5429** -> **0.5096**). The presence false negatives remain stubborn (77 cases) because the context/phase margin features are highly correlated between true and missing pairs.

### Experiment D: Local Pose Refinement
- Limited to small local subpixel searches using the base Correlator.
- Median runtime: **4.34s** (comfortably below the 5.0s limit).

### Experiment E: Calibration
- Built OOF confidence calibration optimizing for final correctness. The AUC dropped because of the ranking instability.

### Experiment F: RGB Bonus Adapter
- Script gb_adapter.py created to provide robust gray consensus without CNNs. Ready for Set D deployment once the 20 optical pairs are released.

## PROMOTION GATE STATUS
⚠️ **REJECTED**: V26 did not surpass V25 Base and caused a localization regression. V25 remains the protected fallback.
