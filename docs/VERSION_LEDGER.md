# DRIFT-SENSE++ MASTER VERSION LEDGER & SCORECARD

This document serves as the single source of truth for the entire engineering history of the Drift-Sense SEM-to-Optical Alignment System.

---

## 1. Official Competition Scoring System (100 Points Budget)

| Dimension | Points | Description & Evaluation Rules |
| :--- | :---: | :--- |
| **Localization Accuracy** | **40** | Weighted score ($0.45 \times \text{SetA} + 0.55 \times \text{SetB}$). Credit tiers: $\le 1\text{px}: 100\%$, $\le 2\text{px}: 80\%$, $\le 3\text{px}: 60\%$, $\le 5\text{px}: 40\%$, $>5\text{px}: 0\%$. |
| **Pose Recovery** | **20** | Scale ($\le 1\%: 100\%$, $\le 2\%: 50\%$, $\le 5\%: 25\%$) + Rotation ($\le 0.25^\circ: 100\%$, $\le 0.50^\circ: 50\%$, $\le 1.0^\circ: 25\%$). Scored **only** when localization is correct ($\le 5\text{px}$). |
| **Absence Rejection** | **15** | Set C absence discrimination (40 absent pairs with same die architecture). Scored on $F_1$ score ($2 \times \frac{P \times R}{P + R}$). |
| **Confidence Calibration** | **10** | Monotonicity of `score` column vs prediction correctness evaluated via ROC-AUC. |
| **Computational Efficiency** | **5** | Hard requirement: median runtime $\le 5.0\text{s/pair}$, hard timeout at 20.0s. |
| **Generator & Reproducibility**| **10** | Deterministic standalone CLI, clean citations, comprehensive failure analysis. |
| **Total Base Score** | **100** | **Championship Target: $\ge 89 - 95 / 100$** |
| **Bonus (RGB Cross-Modal)** | **+10** | Optical RGB cross-spectral assistance (must not degrade grayscale baseline). |

---

## 2. Complete Version Evolution (V1 through V19)

| Version | Main Objective | Key Architectural Innovation | Weighted Loc / Top-1 | Rejection $F_1$ | GT Pool Recall | Runtime (s/pair) | Key Findings & Decisions |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **V1–V5** | Initial FFT & Spatial Baselines | Standard 2D cross-correlation, basic peak picking | 12.40% | 0.0500 | ~25% | 1.2s | Suffered catastrophic failure on scale/rotation mismatches. |
| **V6–V9** | Joint Pose & Coarse Search | Joint scale/rotation grid sweeps | 26.11% | 0.1200 | ~35% | 8.5s | Grid explosions; high latency; periodic clone trapping. |
| **V10–V12** | Decoupled Sequential Pose | Scale-first coarse-to-fine search followed by 1D rotation fine tuning | 35.44% | 0.1905 | 45.0% | 2.1s | Decoupled pose solved rotation/scale trapping; discovered 32.9% periodic density cap bottleneck. |
| **V13** | Density Rescue Exploration | Flat multi-pose candidate unions | 31.20% | 0.1800 | 48.0% | 4.8s | **REJECTED**: Bad-pose noise peaks diluted candidate quality. |
| **V14** | CAR & V14-P1 Multi-Evidence | Confidence-Adaptive Ranking (CAR) + Replica Family Clustering + V14-P1 Gate ($t=0.58$) | 48.88% | 0.3862 | 50.0% | 3.2s | **FROZEN CANDIDATE**: Set B jumped to 57.14% (0.74 px median); periodic failures fell from 67 to 36. |
| **V14-R2**| Linear Context Ranker | Unconditional linear combination of context & phase | 8.43% | — | — | 3.5s | **REJECTED**: Destructive to degraded periodic patterns. |
| **V15** | Forensic Oracle Audit | Upper-bound forensic ceiling analysis on 140 present cases | — | — | 50.0% (500: 74.3%) | — | **CRITICAL DISCOVERY**: Raw plane contains GT 96.43% of the time, but Top-50 NMS only captures 50.00%. Ranker converts 60% of available GT. |
| **V16** | Retrieval 2.0 (Akhilesh Sprint)| Bounded Rescue Queue: fast $K=200$ raw NMS + lightweight context filter returning Top-50 | **50.81%** | 0.3862 | 50.71% | 5.8s | **FROZEN CONTROL (`results/CONTROL_V16/`)**: Set B reached **60.00%** (0.70 px median); broke 50% Top-50 ceiling; periodic failures dropped from 36 to 35. |
| **V17** | Replica Forensics (Audit 35) | Full feature extraction on 35 periodic failure cases ($C_{GT}, C_1, C_2, C_3$) | — | — | — | — | **DECISIVE PHYSICAL FINDING**: 100% of failures attributed: 18 Retrieval Caps (51.4%), 8 Center Drift Bias (22.9%), 6 Symmetry (17.1%), 2 Boundary (5.7%), 1 Noise (2.9%). Proved GT center distance $\mu=119.2\text{px}$ vs False Winner $\mu=245.0\text{px}$ ($p=0.0029$). |
| **V18** | Replica Discriminator 2.0 (Akhilesh)| Periodicity-Adaptive Center Prior + Context + Phase Gate ($V18\_C$) | **Top-1: 23.21%** (Cond: 46.48%) | — | — | <0.01s (eval) | **VALIDATED WINNER**: Conditional Top-1 jumped from 30.99% $\to$ **46.48%** (+50% relative gain over V16 control). Top-1 hits increased from 22 $\to$ 33. |
| **V20** | Presence & Absence Engine (Scalar) | Handcrafted features (margin, cut distance) + Logistic Regression | — | 0.708 | — | — | **REJECTED**: Structural-anchor evidence failed to generalize. F1=0.708, Set C FPR=0.77. Scalar evidence is insufficient for presence discrimination. |

---

## 3. Current 100-Point Score Accounting

$$\text{Frozen Control Baseline (V16)} = \mathbf{61.11 / 100}$$

```text
100-POINT COMPETITION SCORE GAP & TARGETS
┌────────────────────────────────────────────────────────────┐
│ Localization:  20.32 / 40 pts   (Gap: +13.7 to +17.7 pts)  │
│ Pose:          16.50 / 20 pts   (Gap: +1.5 to +3.5 pts)    │
│ Rejection:      5.79 / 15 pts   (Gap: +7.2 to +9.2 pts)    │
│ Calibration:    4.00 / 10 pts   (Gap: +5.0 to +6.0 pts)    │
│ Efficiency:     4.50 / 5 pts    (Gap: +0.5 pts)            │
│ Generator/Doc: 10.00 / 10 pts   (Gap: 0.0 pts)             │
│ TOTAL:         61.11 / 100 pts  (CHAMPIONSHIP TARGET: 89+) │
└────────────────────────────────────────────────────────────┘
```

---

## 4. Active Strategic Roadmap (V18 $\to$ V26)

```text
                  V17 FORENSICS (COMPLETE)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   V19 CANDIDATE RESCUE              V18 REPLICA RANKER
   (Aashish Main Track) - COMPLETE   (Akhilesh Specialist) - COMPLETE
   Recall: 50.7% -> 67.1% (+32.4%)   Cond Top-1: 31.0% -> 46.5% (+50.0%)
            │                                 │
            └────────────────┬────────────────┘
                             ▼
                    V20 SCALAR PRESENCE (REJECTED)
                    F1=0.708, FPR=0.77. Scalar evidence insufficient.
                             │
                             ▼
                    V20.2 PATCH VERIFIER (REJECTED)
                    F1=0.590, Recall=0.506. Local visual evidence insufficient.
                             │
                             ▼
                    V20.3 GLOBAL VALIDATOR
                    Frequency/Phase Domain Validator
                             │
                             ▼
                    V20.4 PRESENCE (FREEZE)
                             │
                             ▼
                    V21 JOINT INTEGRATION
                   Combine V19 Extractor + V18 Ranker + V20.4 Presence
                   Target: Weighted Loc >= 60%, F1 >= 0.85
                             │
                             ▼
                    V22 CALIBRATION & AUC
                             │
                             ▼
                    V23 SUBPIXEL METROLOGY (<=1px)
                             │
                             ▼
                    V24 EFFICIENCY (Median < 5s)
                             │
                             ▼
                    V25 RGB OPTICAL BONUS (+10)
                             │
                             ▼
                    V26 FINAL REFEREE PACKAGE
```
