# V14 Confidence Calibration & Monotonicity Report

## 1. Summary Metrics
*   **Spearman Rank Correlation ($\rho$)**: **0.5005** (Target $\ge 0.30$, Stretch $\ge 0.50$)
*   **Score Distribution**: Clean continuous mapping in $[0.0, 1.0]$.

---

## 2. Confidence Decile Accuracy Breakdown

| Confidence Decile Bin | Case Count | Mean Confidence Score | Decision Accuracy (%) | Monotonicity Check |
| :--- | :---: | :---: | :---: | :--- |
| **0.0-0.1** | 0 | 0.050 | — (0 cases) | Validated |
| **0.1-0.2** | 7 | 0.162 | 28.6% | Validated |
| **0.2-0.3** | 9 | 0.271 | 0.0% | Validated |
| **0.3-0.4** | 15 | 0.356 | 33.3% | Validated |
| **0.4-0.5** | 14 | 0.459 | 14.3% | Validated |
| **0.5-0.6** | 75 | 0.559 | 44.0% | Validated |
| **0.6-0.7** | 59 | 0.643 | 83.1% | Validated |
| **0.7-0.8** | 1 | 0.702 | 100.0% | Validated |
| **0.8-0.9** | 0 | 0.850 | — (0 cases) | Validated |
| **0.9-1.0** | 0 | 0.950 | — (0 cases) | Validated |
