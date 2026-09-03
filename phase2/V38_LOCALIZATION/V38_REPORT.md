# V38 Local Pose Optimization Report

## Runtime Summary
- **Total Pairs**: 180
- **Total Time**: 438.20s
- **Avg Time/Pair**: 2.434s

## Benchmark Output
```

=================================================================
           DRIFT-SENSE++ PHASE 2 HARDENED BENCHMARK
=================================================================
Total Evaluated Pairs: 180
  - Set A (Nominal):   70
  - Set B (Degraded):  70
  - Set C (Absent):    40
-----------------------------------------------------------------
1. LOCALIZATION METRICS (<= 5 px Target):
  Set A <= 1 px: 50.00% | Set A <= 5 px: 58.33% | Median: 2.00 px
  Set B <= 1 px: 31.03% | Set B <= 5 px: 31.03% | Median: 40.41 px
  OFFICIAL WEIGHTED LOC SCORE (0.45*A + 0.55*B): 43.32%
-----------------------------------------------------------------
2. POSE RECOVERY METRICS:
  Set A Scale MAE: 0.0464 | Rotation MAE: 0.0749°
  Set B Scale MAE: 0.0462 | Rotation MAE: 0.0872°
-----------------------------------------------------------------
3. ABSENCE REJECTION METRICS (Set C Target F1 > 0.90):
  Overall Precision: 0.2788 | Recall: 0.7250
  Set C Rejection F1 Score: 0.4028
-----------------------------------------------------------------
4. CONFIDENCE MONOTONICITY:
  Spearman Rank Correlation (rho): 0.1327
-----------------------------------------------------------------
5. FAILURE TAXONOMY SUMMARY:
  - PRESENCE_FALSE_NEGATIVE: 75 cases (41.7%)
  - PERIODIC_REPLICA: 35 cases (19.4%)
  - REJECTION_SUCCESS: 29 cases (16.1%)
  - SUBPIXEL_SUCCESS: 27 cases (15.0%)
  - ABSENCE_FALSE_POSITIVE: 11 cases (6.1%)
  - IN_BOUNDS_SUCCESS: 3 cases (1.7%)
=================================================================

Failure taxonomy written to phase2/V38_LOCALIZATION\failure_taxonomy.csv

```
