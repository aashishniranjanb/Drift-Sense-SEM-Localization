# Phase V20.2: Patch-Level Learned Match Verifier

## Scientific Question
> **Can a lightweight learned model distinguish a genuine reference/search correspondence from a visually convincing periodic replica when given the reference patch and candidate search neighborhood directly?**

## Rationale
V20 decisively proved that scalar features (NCC, margin, distance) compress away the spatial relationship needed to validate degraded matches. We are shifting from "asking a scalar if it's real" to "letting a neural network look at the actual visual evidence."

## Model Ladder
- **V20.2-A (Classical Baselines)**: Measure NCC, SSIM, gradient similarity, phase correlation, and multiscale patch similarity for a sanity baseline.
- **V20.2-B (Siamese CNN)**: Lightweight shared encoder computing distance/similarity between reference crop and search candidate crop.
- **V20.2-C (Two-Stream CNN)**: Independent encoders with late fusion (concat, diff, product) into MLP.
- **V20.2-D (CNN + Handcrafted)**: CNN embeddings concatenated with V20 scalar evidence.
- **V20.2-E (Hard-Negative Mining)**: Crucial phase. Generate negatives directly from V18/V19 failure modes (PERIODIC_REPLICA, HIGH_NCC_WRONG, V18_WRONG_WINNER).
- **V20.2-F (Frozen Verifier)**: Final frozen evaluation.

## Anti-Leakage Constraints
- Strict pair/structure-aware train/val/test splits. Transformations of a test pair must NEVER appear in training.
- V18-C and V19 controls remain IMMUTABLE.

## Acceptance Gate
- Primary Target: $F_1 \ge 0.90$
- PRESENT Recall $\ge 0.95$ (acceptable Set A/B degradation)
- Hard-Negative Rejection Rate: Must explicitly reject the specific candidates that currently fool V18/V19 (Set C FPR << 0.77).
- ROC-AUC $\ge 0.90$
- No test leakage.
