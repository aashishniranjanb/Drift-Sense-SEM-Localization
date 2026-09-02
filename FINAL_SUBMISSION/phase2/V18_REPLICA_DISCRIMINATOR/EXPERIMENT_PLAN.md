# Phase V18: Experiment Plan

## Protocol
1. **Target Population**: All 140 present cases in `data/phase2_dev/pairs.csv`.
2. **Evaluation Metrics**:
   - `Conditional Top-1 Accuracy`: $\% \text{ of cases where } GT \text{ was inside Top-50 candidate pool and was ranked \#1}$.
   - `Absolute Top-1 Accuracy`: $\% \text{ of all 140 cases correctly ranked \#1}$.
   - `Set A Loc (\le 5 px)`: Must not regress from V16 baseline (39.58%).
   - `Set B Loc (\le 5 px)`: Must improve beyond V16 baseline (60.00%).
   - `Weighted Localization Score`: $0.45 \times A + 0.55 \times B$ (Target $\ge 55\%$).

3. **Experiment Ladder Variants**:
   - **V18-A**: $S = NCC - w_{\text{fam}} \cdot (d_{\text{center}}/250)^2$
   - **V18-B**: $S = NCC + 0.15 \cdot \text{Context}_{128} - w_{\text{fam}} \cdot (d_{\text{center}}/250)^2$
   - **V18-C**: $S = NCC + 0.15 \cdot \text{Context}_{128} - 0.20 \cdot \text{PhasePenalty} - w_{\text{fam}} \cdot (d_{\text{center}}/250)^2$
   - **V18-D**: Full Handcrafted Normalized Composite (incorporating PSR, edge distance, and family density).
   - **V18-E**: Calibrated Linear / Logistic Ridge Ranker.
