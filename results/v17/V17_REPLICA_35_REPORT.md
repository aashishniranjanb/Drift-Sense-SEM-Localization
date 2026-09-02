# V17 Replica Discrimination Forensic Report (35 Periodic Failure Audit)

**Total Periodic-Replica Failures Audited:** 35

**Failures with GT Present in Top-50 Pool:** 17 / 35 (48.6%)

**Failures with GT Missing from Top-50 Pool (Retrieval Caps):** 18 (51.4%)


## Statistical Feature Separability (GT vs Winning False Replica)

Analyzing cases where the True GT candidate was inside the Top-50 pool but lost to a false replica at Rank #1:


| Feature | GT Mean | Winner Mean | Mean Diff (GT - Win) | GT Win-Rate (%) | T-Stat | P-Value | Direction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `corr_score` | 0.8328 | 0.8436 | -0.0107 | 94.1% | -4.01 | 0.0010 | GT Lower |
| `psr` | 2.5917 | 2.6190 | -0.0273 | 82.4% | -2.29 | 0.0362 | GT Lower |
| `phase_residual` | 0.0982 | 0.0754 | +0.0229 | 58.8% | 2.77 | 0.0137 | GT Higher |
| `phase_penalty` | 0.0562 | 0.0689 | -0.0126 | 94.1% | -1.56 | 0.1392 | GT Lower |
| `context_32` | 0.7063 | 0.7407 | -0.0344 | 58.8% | -0.85 | 0.4068 | GT Lower |
| `context_64` | 0.7463 | 0.7627 | -0.0164 | 70.6% | -0.81 | 0.4298 | GT Lower |
| `context_128` | 0.7300 | 0.7638 | -0.0338 | 88.2% | -1.65 | 0.1178 | GT Lower |
| `context_combined` | 0.7318 | 0.7588 | -0.0270 | 94.1% | -2.94 | 0.0095 | GT Lower |
| `ssd` | 2095.7505 | 1946.4598 | +149.2907 | 52.9% | 1.66 | 0.1172 | GT Higher |
| `dist_to_center` | 119.2142 | 244.9788 | -125.7646 | 88.2% | -3.50 | 0.0029 | GT Lower |
| `nearest_edge_dist` | 1.2314 | 1.1212 | +0.1102 | 17.6% | 0.96 | 0.3494 | GT Higher |
| `nearest_cut_dist` | 2.1253 | 2.2635 | -0.1383 | 88.2% | -1.43 | 0.1712 | GT Lower |
| `row_spacing` | 5.1765 | 19.5882 | -14.4118 | 76.5% | -0.97 | 0.3443 | GT Lower |
| `col_spacing` | 30.7647 | 19.0000 | +11.7647 | 35.3% | 1.37 | 0.1911 | GT Higher |
| `local_density` | 98.9638 | 100.7915 | -1.8278 | 76.5% | -1.23 | 0.2364 | GT Lower |
| `family_population` | 16.1765 | 17.7647 | -1.5882 | 88.2% | -0.99 | 0.3349 | GT Lower |

## Key Forensic Findings & Conclusions

### Top 5 Strongest Replica Discriminators:

1. **`corr_score`**: Mean Diff = -0.0107, Win-Rate = 94.1%, p = 1.0113e-03 (GT Lower)
2. **`dist_to_center`**: Mean Diff = -125.7646, Win-Rate = 88.2%, p = 2.9345e-03 (GT Lower)
3. **`context_combined`**: Mean Diff = -0.0270, Win-Rate = 94.1%, p = 9.5452e-03 (GT Lower)
4. **`phase_residual`**: Mean Diff = +0.0229, Win-Rate = 58.8%, p = 1.3717e-02 (GT Higher)
5. **`psr`**: Mean Diff = -0.0273, Win-Rate = 82.4%, p = 3.6218e-02 (GT Lower)
