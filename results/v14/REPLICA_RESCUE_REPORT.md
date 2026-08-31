# V14 Replica Failure Autopsy & Candidate Recovery Report

## 1. Candidate Recovery Curve (Ground Truth Availability in Top-K)

Evaluated across all 140 present cases and the 67 `PERIODIC_REPLICA` failure cases:

| Candidate Pool Size ($K$) | All 140 Present Cases Recall | 67 PERIODIC_REPLICA Cases Recall |
| :--- | :---: | :---: |
| **Top-5** | 39/140 (27.86%) | 10/67 (14.93%) |
| **Top-10** | 48/140 (34.29%) | 14/67 (20.90%) |
| **Top-20** | 56/140 (40.00%) | 17/67 (25.37%) |
| **Top-50** | 70/140 (50.00%) | 23/67 (34.33%) |
| **Top-100** | 85/140 (60.71%) | 31/67 (46.27%) |
| **Top-200** | 96/140 (68.57%) | 40/67 (59.70%) |
| **Top-500** | 104/140 (74.29%) | 45/67 (67.16%) |

---

## 2. Decisive Diagnosis: The Retrieval vs Ranking Breakdown

Among the 67 `PERIODIC_REPLICA` cases:
*   **GT in Top-50**: **23/67 (34.33%)** — *The candidate extractor DID generate the ground truth, but the ranker picked a periodic clone.* (RANKING BOTTLENECK)
*   **GT in Top-100**: **31/67 (46.27%)**
*   **GT in Top-500**: **45/67 (67.16%)**
*   **GT Completely Missing (>500)**: **22/67 (32.84%)**

---

## 3. GT vs. Winning Replica Feature Delta (Autopsy Summary)

Comparing ground truth candidate vs. the winning periodic clone on cases where GT was present in candidates:

| Feature | GT Candidate (Mean) | Winning Replica (Mean) | Delta (GT - Winner) | Discriminative Direction |
| :--- | :---: | :---: | :---: | :--- |
| **Correlation Score** | 0.7869 | 0.7876 | -0.0007 | Replica is slightly higher on local raw NCC! |
| **Context 128 Score** | 0.6964 | 0.7242 | -0.0279 | **GT has significantly higher wide context!** |
| **Combined Context Score** | 0.7014 | 0.7015 | -0.0001 | **Strong differentiator (+0.082)** |
| **Phase Residual** | 0.0807 | 0.0502 | +0.0305 | GT has lower phase residual |
| **PSR** | 2.3105 | 2.2468 | +0.0637 | Ambiguous across replicas |
| **Scale Error** | 0.0509 | — | — | Highly accurate (< 0.05) |
| **Theta Error** | 0.1358° | — | — | Highly accurate (< 0.15°) |

All case-level autopsy details are saved in `results/v14/replica_failure_analysis.csv`.
