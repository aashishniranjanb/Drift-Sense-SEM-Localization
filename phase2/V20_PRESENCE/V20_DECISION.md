# V20 DECISION

## Best Feature Set
G2 (nearest_cut_dist inverse) combined with peak_margin and family_population provided the best theoretical separation during forensics. However, on the authoritative phase2 dataset, standardizing and combining them yielded limited success.

## Best Model
Logistic Regression (V20-H). While it performed better than baseline handcrafted thresholds, it still fell short of the 0.90 F1 target.

## Frozen Threshold
Threshold selected was 0.46 on validation.

## Confusion Matrix & Breakdown
- Set A Accuracy: ~72%
- Set B Accuracy: ~65%
- Set C Accuracy: ~23% (High False Positive Rate remains)

## Remaining Mechanisms
- False Positives: Periodic hard negatives in Set C still dominate. The structural anchor features (
earest_cut_dist) are noisy in degraded conditions and sometimes falsely anchor.
- False Negatives: True present cases (Set B degraded) are frequently rejected because the structural anchor is obscured by noise, causing 
earest_cut_dist to fluctuate and trigger the penalty.

## KEEP / MODIFY / REJECT
**REJECT**. The Structural-Anchor hypothesis, while physically sound on clean subsets, fails to generalize robustly across the entire authoritative dataset (F1 < 0.90, poor recall on degraded, poor specificity on hard negatives).

## Recommendation for V21
The current feature space (correlation, PSR, context, structural anchors) is insufficient to resolve the periodic ambiguity without destroying recall on noisy present cases. Reconsider the entire presence formulation before V21. We may need to investigate spatial frequency analysis, PACE re-ranking enhancements, or a purely learned local patch classifier (CNN/ViT) to distinguish genuine structures.
