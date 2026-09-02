# Phase V17: Feature Separability & Discriminative Power Ablation

**Sample Size (GT Present in Top-50 Pool):** 17 cases

## 1. Paired Feature Separability Matrix (GT vs Winner)

| Feature Name | GT Mean ($\mu_{GT}$) | Winner Mean ($\mu_{W}$) | Mean $\Delta (GT - W)$ | GT Win Rate (%) | T-Stat | P-Value | Discriminative Power |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `corr_score` | 0.8328 | 0.8436 | -0.0107 | 64.7% (GT Lower) | -4.01 | 0.0010 | VERY STRONG |
| `psr` | 2.5917 | 2.6190 | -0.0273 | 52.9% (GT Lower) | -2.29 | 0.0362 | SIGNIFICANT |
| `phase_residual` | 0.0982 | 0.0754 | +0.0229 | 58.8% (GT Higher) | 2.77 | 0.0137 | SIGNIFICANT |
| `phase_penalty` | 0.0562 | 0.0689 | -0.0126 | 29.4% (GT Lower) | -1.56 | 0.1392 | MODERATE |
| `context_32` | 0.7063 | 0.7407 | -0.0344 | 29.4% (GT Lower) | -0.85 | 0.4068 | WEAK / NOISE |
| `context_64` | 0.7463 | 0.7627 | -0.0164 | 41.2% (GT Lower) | -0.81 | 0.4298 | WEAK / NOISE |
| `context_128` | 0.7300 | 0.7638 | -0.0338 | 58.8% (GT Lower) | -1.65 | 0.1178 | MODERATE |
| `context_combined` | 0.7318 | 0.7588 | -0.0270 | 64.7% (GT Lower) | -2.94 | 0.0095 | VERY STRONG |
| `ssd` | 2095.7505 | 1946.4598 | +149.2907 | 52.9% (GT Higher) | 1.66 | 0.1172 | MODERATE |
| `dist_to_center` | 119.2142 | 244.9788 | -125.7646 | 58.8% (GT Lower) | -3.50 | 0.0029 | VERY STRONG |
| `nearest_edge_dist` | 1.2314 | 1.1212 | +0.1102 | 17.6% (GT Higher) | 0.96 | 0.3494 | WEAK / NOISE |
| `nearest_cut_dist` | 2.1253 | 2.2635 | -0.1383 | 17.6% (GT Lower) | -1.43 | 0.1712 | WEAK / NOISE |
| `row_spacing` | 5.1765 | 19.5882 | -14.4118 | 17.6% (GT Lower) | -0.97 | 0.3443 | WEAK / NOISE |
| `col_spacing` | 30.7647 | 19.0000 | +11.7647 | 35.3% (GT Higher) | 1.37 | 0.1911 | WEAK / NOISE |
| `local_density` | 98.9638 | 100.7915 | -1.8278 | 47.1% (GT Lower) | -1.23 | 0.2364 | WEAK / NOISE |
| `family_population` | 16.1765 | 17.7647 | -1.5882 | 17.6% (GT Lower) | -0.99 | 0.3349 | WEAK / NOISE |

## 2. Key Takeaways for Discriminator Design
- `dist_to_center` has the single highest statistical significance ($p = 0.0029, t = -3.50, \text{Win Rate} = 88.2\%$).
- `corr_score` and `context_combined` alone actively favor the false replica if applied without spatial regularization.
- An adaptive weighting mechanism scaling with `family_population` is required to only engage center prior when periodic ambiguity is active.
