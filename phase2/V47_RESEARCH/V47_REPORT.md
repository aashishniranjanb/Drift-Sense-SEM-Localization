# V47 RESCUE VALIDATOR REPORT

## Forensic Feature Analysis
Separating true rescues (Pop B) from background false-accepts (Pop C):

| Feature | AUC | Direction | Median B (Real) | Median C (Absent) |
|---|---|---|---|---|
| dist_border | 0.997 | + | 407.75 | 151.25 |
| dist_center | 0.994 | - | 0.225 | 0.806 |
| v25_ml_score | 0.793 | + | 0.176 | 0.000 |
| delta_ncc | 0.732 | + | -0.001 | -0.009 |
| ncc_pct | 0.725 | + | 99.999 | 99.995 |
| dist_v25_v46 | 0.700 | - | 490.98 | 700.34 |
| prom10_ncc | 0.693 | + | 0.746 | 0.683 |

*(Center/border metrics omitted from primary models to avoid safety regression)*

## V47 Scoreboard (Out-of-Fold Evaluation)
Target: Recover V46 candidates without corrupting V25 absent rejection (max 1 FP limit).

| Model | T_absent | T_present | Rescued | Broken | New Absent FP |
|---|---|---|---|---|---|
| Logistic Regression | 0.25 | 0.15 | 18 | 1 | 1 |
| HGB (Depth 2) | 0.10 | 0.10 | 17 | 0 | 1 |
| HGB (Depth 3) | 0.10 | 0.10 | 17 | 0 | 1 |
| Hand Gate | - | - | 0 | 0 | - |

## Final Analysis
By training a shallow, depth-2 HistGradientBoostingClassifier exclusively on orthogonal structural properties (peak prominence, competitor density, localized curvature, percentile scoring) rather than raw 4-signal correlation values, V47 effectively separates genuine targets from periodic background noise.

The HGB-2 model safely recovers **17 genuine candidates** that were missed by V25, breaks **0** protected V25 winners, and introduces only **1** false positive into the rejection pipeline.

**RECOMMENDATION:** PROMOTE
V47 satisfies the highest PROMOTE immediately threshold (>=15 rescues, 0 broken, <=1 new FP). The Desktop localization engine should be stacked and frozen as V47 = V25 + V46_Pool + V47_Gate + V39_Pose. All compute should now transfer to Laptop 1 to attack the final Rejection and Calibration gaps for the 96+ score.
