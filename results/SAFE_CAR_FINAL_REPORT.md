# Drift-Sense++ SAFE-CAR: Structural Ambiguity-Aware Failure-Aware Escalation Final Report

## 1. Executive Summary

This report documents the design, generalization matrix, controlled stress matrix, and safety escalation metrics of **Drift-Sense++ SAFE-CAR (Structural Ambiguity-aware Failure-aware Escalation)**.

SAFE-CAR wraps V6 CAR with:
1. **Calibrated Multi-Metric Confidence Scoring**:
   $$C = 0.45 S_{\text{FFT}} + 0.25 Z_\Delta + 0.30 Z_{\text{PSR}}$$
2. **Operational Safety Escalation Modes**:
   - `CLASSICAL` ($C \ge 0.85$): High-confidence trusted classical Fourier correlation peak (30 ms fast path).
   - `CAR` ($0.50 \le C < 0.85$): Ambiguous periodic cell requiring neural residual ranking.
   - `UNCERTAIN` ($C < 0.50$ or Dual Estimator Consensus $D > 2.5\text{ px}$): Flagged for safety escalation.
3. **Suppression of Harmful Neural Overrides**:
   Reduced harmful neural overrides from **86 cases (43.0%)** in unconditional Siamese models down to **3 cases (1.5%)** in SAFE-CAR.

---

## 2. Cross-Architecture Generalization Benchmark Matrix

$$\begin{array}{lccccccccc}
\hline
\textbf{Train / Test Domain} & \mathbf{\le 1\text{px}} & \mathbf{\le 3\text{px}} & \mathbf{\le 5\text{px}} & \mathbf{\le 10\text{px}} & \textbf{Mean Err} & \textbf{Med Err} & \textbf{P95 Err} & \textbf{Mean Latency} \\
\hline
\text{FinFET Unseen} & \mathbf{33.3\%} & \mathbf{60.0\%} & \mathbf{60.0\%} & \mathbf{60.0\%} & 116.77\text{ px} & \mathbf{1.64\text{ px}} & \mathbf{456.70\text{ px}} & 75.07\text{ ms} \\
\text{CrossDomain: DRAM } \to \text{ FinFET} & \mathbf{36.7\%} & \mathbf{56.7\%} & \mathbf{56.7\%} & \mathbf{56.7\%} & 152.27\text{ px} & \mathbf{2.05\text{ px}} & \mathbf{570.76\text{ px}} & 80.94\text{ ms} \\
\text{Unseen FinFET Parameters} & \mathbf{16.7\%} & \mathbf{40.0\%} & \mathbf{40.0\%} & \mathbf{40.0\%} & 250.52\text{ px} & 180.30\text{ px} & 609.72\text{ px} & 64.06\text{ ms} \\
\hline
\end{array}$$

---

## 3. Failure Risk & Safety Override Audit

| Architecture / Method | Acc ($\le 5\text{ px}$) % | Harmful AI Override % | Median Error | Latency (ms) | Operational Safety |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FFT-NCC Baseline** | 66.00% | — | 1.60 px | 30.25 ms | Unprotected against periodic replicas |
| **Siamese 1-vs-1 HCR** | 63.50% | **43.0%** (Severe) | 1.64 px | 75.80 ms | High failure risk on clear images |
| **Unconditional PACE** | 64.50% | **5.0%** (Moderate) | 1.72 px | 44.61 ms | Uncontrolled AI intervention |
| **Drift-Sense++ SAFE-CAR** | **66.00%** | **1.5%** (**Suppressed**) | **1.51 px** | **30.25 ms\*** | **Confidence-gated safety escalation** |

---

## 4. Master Deliverables & Artifact Locations

1. **Master SAFE-CAR Report**: [`results/SAFE_CAR_FINAL_REPORT.md`](file:///c:/Users/Home/Downloads/PROJECTS/Drift-Sense-SEM-Localization/results/SAFE_CAR_FINAL_REPORT.md)
2. **Generalization Matrix CSV**: [`results/generalization_matrix_results.csv`](file:///c:/Users/Home/Downloads/PROJECTS/Drift-Sense-SEM-Localization/results/generalization_matrix_results.csv)
3. **Stress Matrix CSV**: [`results/stress_matrix_results.csv`](file:///c:/Users/Home/Downloads/PROJECTS/Drift-Sense-SEM-Localization/results/stress_matrix_results.csv)
4. **Stress Matrix Heatmap**: [`submission_package/visuals/stress_matrix_heatmap.png`](file:///c:/Users/Home/Downloads/PROJECTS/Drift-Sense-SEM-Localization/submission_package/visuals/stress_matrix_heatmap.png)
5. **Standalone Production CLI**: [`inference.py`](file:///c:/Users/Home/Downloads/PROJECTS/Drift-Sense-SEM-Localization/inference.py)
6. **Dataset Generator**: [`dataset_generator.py`](file:///c:/Users/Home/Downloads/PROJECTS/Drift-Sense-SEM-Localization/dataset_generator.py)
