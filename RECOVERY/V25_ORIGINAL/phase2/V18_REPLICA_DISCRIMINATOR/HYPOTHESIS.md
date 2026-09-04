# Phase V18: Hypotheses

## Hypothesis H1: Adaptive Spatial Regularization
*Universal center weighting destroys off-center nominal targets in Set A, but modulating the center penalty by periodic family cluster size ($w_{\text{fam}} = f(\text{population})$) will suppress peripheral replicas while leaving isolated patterns untouched.*

## Hypothesis H2: Orthogonal Evidence Fusion
*Combining raw cross-correlation ($NCC$), high-frequency phase consistency ($1 - \text{phase\_res}$), wide structural context ($\text{context}_{128}$), and spatial center priors will break symmetry across all periodic arrays.*

## Hypothesis H3: Learned Linear Consistency
*A calibrated linear ranker trained on pairwise feature deltas ($C_{\text{GT}} - C_{\text{Replica}}$) will discover the optimal Pareto-optimal weighting faster than manual heuristic tuning.*
