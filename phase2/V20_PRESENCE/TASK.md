# Phase V20: Presence & Absence Engine (Shanganidhi Stream)

## Objective
Build a PRESENT / ABSENT engine independently of localization ranking, focusing strictly on resolving false rejections and hard negatives without compromising the V18-C Ranker's gains. 

## Scientific Question
> **Determine whether "Structural anchoring + peak distinctiveness + periodic-family evidence" provides a generalizable PRESENT/ABSENT discriminator.**

## Experiment Ladder (Revised after V20-F Forensics)
- **V20-G (Ablation / Feature Validation)**: Validate the hypothesis that structural anchors (`nearest_cut_dist`) and distinctiveness (`peak_margin`) separate PRESENT from ABSENT. Ablate over combinations G0 to G6. Test scientific transformations (raw, normalized, clipped, decay) rather than hard thresholds.
- **V20-H (Logistic Regression)**: Train a calibrated Logistic Regression model using the strongest validated feature set from V20-G. Fixed train/val/test splits, no per-case tuning, test set evaluated once.

## Constraints & Gates
- **Do not modify** `register.py` or `inference_phase2.py`.
- **Anti-leakage rule**: Strict train/test fixed split. Do not derive the threshold from the final benchmark.
- **Acceptance criteria (Tightened)**: 
  - $F_1 \ge 0.90$
  - PRESENT recall $\ge 0.95$
  - Materially lower Set C FPR than 95.45%
  - No unacceptable Set A/B degradation
  - No test leakage
  - Runtime acceptable

## Key Deliverables (V20-G & V20-H)
1. `results/V20_G_ABLATION.csv`
2. `results/V20_H_LOGREG.csv`
3. `results/V20_H_TEST_PREDICTIONS.csv`
4. `results/V20_H_CONFUSION_MATRIX.csv`
5. `V20_G_ABLATION.md`
6. `V20_H_RESULTS.md`
7. `V20_DECISION.md`
