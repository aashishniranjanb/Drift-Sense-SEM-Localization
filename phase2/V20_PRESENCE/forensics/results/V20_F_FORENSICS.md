# V20-F: Hard-Negative Forensics Report

## Objective
Analyze the physical and evidence characteristics that caused the V20-E classifier to produce 21 false positives (predicting PRESENT for genuine ABSENT hard-negative cases).

## Discriminative Features: PRESENT-DEGRADED vs. ABSENT-HARD-NEGATIVE

Our analysis reveals that relying on standard correlation and local context similarity completely fails to reject periodic hard negatives, because their local patches perfectly match the template structure.

The most discriminative features are:

### 1. Nearest Cut Distance (nearest_cut_dist)
- **PRESENT-Nominal**: ~14.06 (Mean)
- **PRESENT-Degraded**: ~4.48 (Mean)
- **ABSENT (Set C)**: ~27.02 (Mean)
- *Finding*: Hard negatives are found in completely uniform periodic regions, far away from unique structural breaks (cuts). Degraded true positives tend to lock onto structural cuts (low nearest cut distance) because those provide the only unique signal in noise.

### 2. Peak Margin (peak_margin)
- **PRESENT-Nominal**: 0.023
- **PRESENT-Degraded**: 0.027
- **ABSENT (Set C)**: 0.002
- *Finding*: Set C false positives have near-zero peak margin (0.002). This means there are multiple almost identical peaks. The true location (even when degraded) has a 10x higher peak margin.

### 3. Family Population (family_population)
- **PRESENT-Degraded**: 44.6
- **ABSENT (Set C)**: 50.0 (Maxed out)
- *Finding*: All Set C cases hit the maximum threshold for replica families, indicating severe ambiguity.

## Mechanism Categorization
Every single false-positive in Set C was classified under **PERIODIC_REPLICA**. The engine is finding an identical array of fins/gates but failing to realize it is in the wrong part of the chip because the local field is identical and lacks distinguishing features (cuts/edges).

## Conclusion to Scientific Question
*What physical/evidence characteristics distinguish a genuinely present structure from a hard negative that produces a convincing correlation peak?*

A genuine degraded match relies on unique structural features (like gate cuts) leading to a lower 
earest_cut_dist and a higher peak_margin. A hard negative simply matches a periodic lattice in a featureless region (high 
earest_cut_dist, near-zero peak_margin, maxed-out amily_population).
