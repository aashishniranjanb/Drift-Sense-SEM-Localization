# PHASE 2 DATA LEGALITY AUDIT

## 1. Status of data/phase2_dev/pairs.csv
* Dataset size: 180 image pairs (70 Set A, 70 Set B, 40 Set C).
* Purpose in Challenge Rules:
  - In the Phase 2 competition specification, organizer-provided development/validation sets (phase2_dev) are intended for **algorithm validation, threshold tuning, and diagnostic assessment**, NOT as arbitrary supervised training labels for full parameter fitting.
  - While threshold selection and hyperparameter exploration on dev data are permitted, training unconstrained supervised models (such as deep neural networks or dense gradient-boosted trees) directly on all 180 ground-truth labels and evaluating on the same set creates **extreme overfitting** and **severe generalization failure** on the unseen test evaluation.

## 2. Evidence of Overfitting Risk
* In V25, the presence model (presence.pkl) was fitted on all 180 pairs. It attained an apparent Rejection F1 of 0.5429 (8.14/15 pts) and Spearman rank of 0.5490 (5.49/10 pts).
* When evaluated under rigorous 5-fold cross-validation (Out-Of-Fold), the true generalizable performance dropped:
  - Apparent Rejection F1: 0.5429 -> Out-Of-Fold F1: ~0.3881.
  - Apparent Monotonicity: 0.5490 -> Out-Of-Fold Monotonicity: ~0.1505.
* This proves that fitting high-capacity classifiers directly on phase2_dev without OOF validation inflates local scores while creating catastrophic vulnerability on the final blind evaluation set.

## 3. Operational Policy for V27 Final
1. **No full re-training on all 180 labels**: We will NOT ship high-capacity models trained on all 180 labels that risk generalization collapse.
2. **Conservative Confidence Gating**: We only explore conservative threshold modifications, monotonic rank-preserving confidence transforms, and transparent rule-based / low-parameter linear gates.
3. **Protected Localization Engine**: The physical localization engine (template extraction, FFT cross-correlation, subpixel refinement, rotation/scale estimation) is 100% unsupervised/deterministic and completely immune to label-leakage or overfitting.
4. **Promotion Threshold**: If no conservative gating mechanism cleanly beats V25 on both total score and stability without compromising localization, **V25 baseline remains the submitted configuration**.
