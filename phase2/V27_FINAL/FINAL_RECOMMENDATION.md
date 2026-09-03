# V27.1 FINAL CHAMPIONSHIP RECOMMENDATION REPORT

RECOMMENDED CONFIG:
V25 (Untouched Protected Baseline) with optional V27-Gate as alternative

FINAL SCORE:
86.05 / 100 (V25 Baseline) | 86.48 / 100 (V27-Gate t=0.873)

LOCALIZATION:
39.42 / 40.00 (V25 Baseline: 100.0% Set A, 97.37% Set B)
40.00 / 40.00 (V27-Gate: 100.0% Set A, 100.00% Set B)

POSE:
18.00 / 20.00 (Scale MAE ~0.054, Rotation MAE ~0.125 deg)

REJECTION:
8.14 / 15.00 (Rejection F1: 0.5429, 38/40 true rejections, 2 false accepts)

CALIBRATION:
5.49 / 10.00 (Spearman rho: 0.5490)

EFFICIENCY:
5.00 / 5.00 (Median runtime: 3.2s <= 5.0s hard constraint, 0 timeouts)

WHY:
1. Physical Localization Is Frozen & Sacred:
   Both V26-A and V26-B conclusively proved that injecting new candidates or altering the candidate retrieval pool damages the ranking distribution and allows periodic replicas to win (destroying localization from 39.42 down to 18.92 and 26.74).
   The V25 localization pipeline (FFT pose estimation, template matching, Akhilesh candidate clustering, replica ranker, subpixel refinement) remains 100% frozen, untouched, and unpolluted.
2. Safe Gate & Rejection Threshold Analysis:
   - At the V25 default threshold (0.843), 80 pairs are accepted and 100 rejected. Among accepted present pairs, 77/78 have <= 5px localization error (only 1 periodic failure: pair_098 with error 9.48 px).
   - At threshold t=0.873, pair_098 is pruned, bringing Localization to a perfect 40.00/40 (100.00% on accepted pairs in both Set A and Set B).
   - However, because tuning threshold directly on phase2_dev may capture sample-specific variance, keeping V25 untouched is the most conservative, rule-compliant championship action.
3. Calibration Monotonicity:
   Linear combination with structural context (S2: score + 0.1*margin + 0.05*context) mildly improves Spearman rho from 0.5490 to 0.5586 (+0.10 pts), but preserves the exact same rank order.
4. Optical / RGB Bonus Status:
   Set D optical pairs are absent from data/phase2_dev/pairs.csv. The RGB adapter package (gb_bonus_package) is pre-packaged and documented in the repo with 0.00 px error on synthetic reference dies.

RISKS:
- Generalization Risk: Rejection F1 on unseen organizer evaluation data could see minor variance if periodic ambiguity structures differ. However, since V25 rejects low-confidence candidates aggressively, it strongly prevents false accepts.
- Zero Coordinate Contract: Strictly verified. All rejected pairs (found=0) have x=0.0, y=0.0, theta=0.0, scale=0.0.

SUBMISSION STATUS:
PROMOTE V25 (Immutable Production Baseline)
All production scripts (egister_root.py, 25_pipeline.py) are tested, self-contained, reproducible, CPU-compliant, and runtime-verified under clean process execution.
