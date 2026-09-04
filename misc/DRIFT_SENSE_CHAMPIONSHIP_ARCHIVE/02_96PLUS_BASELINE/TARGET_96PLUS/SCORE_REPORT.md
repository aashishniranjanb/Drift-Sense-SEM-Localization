# SCORE_REPORT — TARGET_96PLUS Optimization Target

## Overview
- **Name:** Grayscale Optimization Target
- **Target Raw Score:** 96.000 / 100
- **Anchor Baseline:** V54 Scale Refined Golden Anchor @ 91.040 / 100
- **Status:** OPTIMIZATION TARGET (UNVERIFIED)

## Benchmark Target Metric Summary

```
=================================================================
             TARGET 96+ GRAYSCALE BENCHMARK PROJECTION
=================================================================
Component                |   Baseline 91.040 | Target 96+ Goal
-----------------------------------------------------------------
Localization (40)        |            40.000 |         40.000
Rejection (15)           |             8.028 |         13.000
Pose (20)                |            19.743 |         19.800
Calibration (10)         |             8.269 |          9.200
Efficiency (5)           |             5.000 |          5.000
Documentation (10)       |            10.000 |         10.000
-----------------------------------------------------------------
TOTAL (100)              |            91.040 |         96.040+
=================================================================
```

## Mandatory Verification Protocol
Before renaming `TARGET_96PLUS` to `96PLUS_VERIFIED`:
1. Execute clean-room `register.py --input data/phase2_dev/pairs.csv --output predictions.csv`
2. Run single-file exact competition scorer.
3. Confirm 0 regressions on 76 verified successes.
4. Confirm total score ≥ 96.000.
