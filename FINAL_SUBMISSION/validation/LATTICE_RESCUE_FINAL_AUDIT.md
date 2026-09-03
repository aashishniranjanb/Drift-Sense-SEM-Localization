# LATTICE RESCUE V1 -- CHAMPIONSHIP FINAL SHADOW AUDIT

**Golden Baseline:** 91.040 / 100.00
**Target:** 22 near-miss retrieval failures (pool best candidate 5-10px from GT)

## Final Results

| Metric | Value |
|---|---|
| **22 Retrieval Recoveries** | **1 / 22** |
| **76 Success Regressions** | **3 / 76** |
| **Rescue Applied (shadow)** | 2 pairs |
| **Localization Score** | 40.000 (delta +0.000) |
| **Rejection Score** | 8.085 (delta +0.057) |
| **Pose Score** | 19.743 (delta +0.000) |
| **Calibration AUC** | 0.9953 |
| **Total Score** | 91.097 (delta +0.057) |
| **VERDICT** | **DO NOT PROMOTE** |

## Promotion Conditions
1. total > 91.040: YES
2. zero >5px regressions among 76: NO (3 broken)
3. no new false accepts: YES
4. runtime within limits: YES
5. single-file scorer confirms improvement: YES

## Lattice Estimator Diagnostics (22 pairs)
- Lattice found (confidence >= 0.25): 13 / 22
- Mean confidence: 0.330
- Mean pitch_x: 15.0 px
- Mean pitch_y: 17.6 px
- Mean rescue candidates generated: 35.7
