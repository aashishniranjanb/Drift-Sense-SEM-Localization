# Phase V17: Hypotheses

## Hypothesis H1: Center Drift Prior Dominance
*In SEM alignment navigation, optical-to-SEM drift follows an approximately Gaussian spatial distribution centered near the search FOV origin. Peripheral periodic clones on die boundaries artificially achieve higher correlation due to high-contrast structural edges.*
- **Prediction**: True GT candidates will have significantly lower Euclidean distance to search center ($\mu_{GT} \ll \mu_{Winner}$).

## Hypothesis H2: High-Frequency Phase Cancellation
*In periodic DRAM/FinFET arrays, false replicas exhibit severe subpixel phase distortion when correlated against the reference template, leading to elevated phase residuals compared to true instances.*
- **Prediction**: True GT candidates will show sharper subpixel phase alignment than peripheral noise peaks.

## Hypothesis H3: Multi-Scale Contextual Consistency
*Local correlation (32x32) cannot differentiate repetitive cells, but wide contextual neighborhoods (128x128) encompass non-periodic die features (guard rings, power rails, gate cuts) that break replica symmetry.*
- **Prediction**: Multi-scale context will distinguish replicas when wide non-periodic features exist in the search FOV.
